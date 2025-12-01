import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, UNet_conditional_ori, UNet_conditional_deep, UNet_conditional_sa_5, UNet_conditional_dc_5, EMA
from modules_2 import UNet_gemini
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from pytorch_msssim import ssim
from dataclasses import asdict
import timm
import torch.nn.functional as F
import albumentations as A
import cv2
import lpips
import itertools
import wandb
from make_dataset import DatasetDigitalStaining
from ddpm_conditional import Diffusion_ddim, Diffusion_ddpm
from config import DDPMConfig
from vae_model import VAE_model


def tensor_ssim(img1, img2):
    # return 1.0 - ssim(img1, img2, data_range=1.0, size_average=True)
    return 1.0 - ssim(img1, img2, data_range=2.0, size_average=False)

def create_gaussian_blending_mask(patch_size, device):
    """
    中央が1に近く、端が0に近いガウシアン風の重み付けマップを生成する。
    PyTorch Tensorとして作成し、GPU上に配置する。
    """
    x_coords = torch.arange(patch_size, device=device)
    y_coords = torch.arange(patch_size, device=device)
    center = (patch_size - 1) / 2.0

    dist_from_center_x = (x_coords - center) ** 2
    dist_from_center_y = (y_coords - center) ** 2

    sigma = patch_size / 4.0
    mask_x = torch.exp(-dist_from_center_x / (2 * sigma ** 2))
    mask_y = torch.exp(-dist_from_center_y / (2 * sigma ** 2))

    # 2Dの重みマップ (1, 1, H, W) の形式にしてブロードキャスト可能にする
    blending_mask = torch.outer(mask_y, mask_x).unsqueeze(0).unsqueeze(0)

    return blending_mask

class DDPM(nn.Module):
    def __init__(self, config: DDPMConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        # --- パスとデータに関する設定 ---
        self.dir = self.config.dir
        self.train_folders = self.config.train_folders
        self.val_folders = self.config.val_folders
        self.test_folders = self.config.test_folders
        self.images_to_use = self.config.images_to_use

        # --- 実験管理 (W&Bなど) ---
        self.name = self.config.name
        self.group = self.config.group

        # --- 学習ループに関する設定 ---
        self.n_epoch = self.config.n_epoch
        self.learning_rate = self.config.learning_rate
        self.batch_size = self.config.batch_size
        self.patches_per_epoch = self.config.patches_per_epoch
        self.patches_per_epoch_val = self.config.patches_per_epoch_val
        self.val_epoch = self.config.val_epoch
        self.w_ssim = self.config.w_ssim
        self.w_target_ssim = self.config.w_target_ssim
        self.w_target_lpips = self.config.w_target_lpips
        self.lr_epoch = self.config.lr_epoch

        # --- UNetモデルのアーキテクチャ設定 ---
        self.model_unet = self.config.model_unet
        self.in_chans = self.config.in_chans
        self.dim_mults = self.config.dim_mults
        self.real_pred = self.config.real_pred

        # --- 拡散過程に関する設定 ---
        self.noise_steps = self.config.noise_steps
        self.noise_add = self.config.noise_add

        # --- 推論に関する設定 ---
        self.img_size = self.config.img_size
        self.pred_size = self.config.pred_size
        self.num_inference_steps = self.config.num_inference_steps
        self.mode_dif = self.config.mode_dif
        self.cfg_scale = self.config.cfg_scale
        self.no_label = self.config.no_label
        self.val_crop = self.config.val_crop

        # --- 環境設定 ---
        self.num_workers = self.config.num_workers
        self.device = self.config.device

        # --- VAE ---
        self.vae_pth = self.config.vae_pth
        self.latent_dim = self.config.latent_dim
        self.hidden_dims = self.config.hidden_dims

        c_out = 8 if self.real_pred else 4
        self.in_chans = (self.in_chans + 1) * 4
        if self.model_unet == "original":
            self.model = UNet_conditional_ori(
                c_in=self.in_chans,
                c_out=c_out,
                device=self.device
            ).to(self.device)
        elif self.model_unet == "deep":
            self.model = UNet_conditional_deep(
                c_in=self.in_chans,
                c_out=c_out,
                device=self.device
            ).to(self.device)
        elif self.model_unet == "dc_5":
            self.model = UNet_conditional_dc_5(
                c_in=self.in_chans,
                c_out=c_out,
                device=self.device
            ).to(self.device)
        elif self.model_unet == "sa_5":
            self.model = UNet_conditional_sa_5(
                c_in=self.in_chans,
                c_out=c_out,
                device=self.device
            ).to(self.device)
        elif self.model_unet == "gemini":
            self.model = UNet_gemini(
                c_in=self.in_chans,
                c_out=c_out,
                dim_mults=self.dim_mults,
                device=self.device
            ).to(self.device)
        else:
            self.model = UNet_conditional(
                c_in=self.in_chans,
                c_out=c_out,
                device=self.device
            ).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        if self.mode_dif == "ddim":
            self.diffusion = Diffusion_ddim(noise_steps=self.noise_steps, device=self.device,
                                            noise_add=self.noise_add, cfg_scale=self.cfg_scale)
        # else:
        #     self.diffusion = Diffusion_ddpm(noise_steps=self.noise_steps, img_size=self.img_size, device=self.device,
        #                                     noise_add=self.noise_add, cfg_scale=self.cfg_scale)

        self.vae = VAE_model(in_channels=1, latent_dim=self.latent_dim, hidden_dims=self.hidden_dims).to(self.device)
        self.vae.load_state_dict(torch.load(f"{self.vae_pth}", map_location=self.device))
        self.vae.eval()  # 推論モード
        for param in self.vae.parameters():
            param.requires_grad = False  # VAEは学習しない

        self.weight = 1.0
        self.ema = EMA(0.995)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.loss_fn_mse = nn.MSELoss().to(self.device)
        self.loss_fn_ssim = tensor_ssim
        self.loss_fn_lpips = lpips.LPIPS(net='alex').to(self.device)
        self.hist = {"train": [], "val": [], "test": []}
        self.val_list = [
            "ssim",
            "lpips",
            "ssim_ema",
            "lpips_ema",
        ]
        self.best_score_list = [float('inf')] * len(self.val_list)
        self.min_epoch_list = [0] * len(self.val_list)
        self.epoch = 0

        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        # 最終的な画像サイズ
        final_size = self.img_size
        # 最初に切り出す、一回り大きいサイズ
        initial_crop_size = self.img_size * 2
        augment = A.Compose(
            [
                # 1. 元画像から大きめにクロップ (128x128)
                A.RandomCrop(height=initial_crop_size, width=initial_crop_size, border_mode=cv2.BORDER_REFLECT_101),
                # 2. そのパッチに対して回転・拡縮
                #    ※拡縮しすぎると中央に黒い部分が入る可能性があるので、scale_limitは控えめに
                A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=30, p=1.0),
                # 3. 変換されたパッチの中央から最終サイズをクロップ (64x64)
                A.CenterCrop(height=final_size, width=final_size),
            ],
            additional_targets={"image1": "image",
                                "image2": "image"},
            strict=True,
            seed=137,
        )

        # Train DataLoader
        img_folders = [os.path.join(self.dir, f) for f in self.train_folders]
        datasets = [DatasetDigitalStaining(img_folders[i], augmentation=augment) for i in range(len(self.train_folders))]
        combined_dataset = ConcatDataset(datasets)
        self.batch_size_train = self.batch_size
        self.batch_size = 1
        self.train_loader = DataLoader(combined_dataset, batch_size=self.batch_size_train, shuffle=True,
                                       num_workers=self.num_workers,
                                       pin_memory=True, persistent_workers=self.num_workers > 0)

        # Validation DataLoader
        img_folders = [os.path.join(self.dir, f) for f in self.val_folders]
        datasets = [DatasetDigitalStaining(img_folders[i], augmentation=None) for i in range(len(self.val_folders))]
        combined_dataset = ConcatDataset(datasets)
        self.val_loader = DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=False,
                                     num_workers=self.num_workers, drop_last=True,
                                     pin_memory=True, persistent_workers=self.num_workers > 0)

        # Test DataLoader
        img_folders = [os.path.join(self.dir, f) for f in self.test_folders]
        datasets = [DatasetDigitalStaining(img_folders[i], augmentation=None) for i in range(len(self.test_folders))]
        combined_dataset = ConcatDataset(datasets)
        self.test_loader = DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers, drop_last=True,
                                      pin_memory=True, persistent_workers=self.num_workers > 0)

    def all(self):
        self.train()
        self.test()

    def _wandb_init(self):
        self.run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="kohei_tokyo-the-university-of-tokyo",
            # Set the wandb project where this run will be logged.
            project="Diffusion_model",
            name=self.name,
            group=self.group,
            # Track hyperparameters and run metadata.
            config=asdict(self.config),
        )

    def train(self, noise_add=None, cfg_scale=None, mode_dif=None):
        if cfg_scale is not None:
            self.cfg_scale = cfg_scale
        if noise_add is not None:
            self.noise_add = noise_add
        if mode_dif is not None:
            self.mode_dif = mode_dif
        self._wandb_init()

        for self.epoch in range(self.n_epoch):
            print(f"Epoch {self.epoch + 1}/{self.n_epoch}")
            self.calc_epoch("train")
            if self.epoch % self.val_epoch == 0:
                self.calc_epoch("val")
                self.show_result("val")
        for val_n in range(len(self.val_list)):
            print(f"min_epoch_{self.val_list[val_n]} {self.min_epoch_list[val_n]}")
        min_epoch_data = [[k, v] for k, v in zip(self.val_list, self.min_epoch_list)]
        wandb.log({f"test/min_epoch": wandb.Table(data=min_epoch_data, columns=["index", "epoch"])})

    def test(self, noise_add=None, cfg_scale=None, mode_dif=None):
        if cfg_scale is not None:
            self.cfg_scale = cfg_scale
        if noise_add is not None:
            self.noise_add = noise_add
        if mode_dif is not None:
            self.mode_dif = mode_dif
        test_list = ["final", "ssim", "lpips"]
        for self.test_id in test_list:
            print(f"Test {self.test_id}")
            self.model.load_state_dict(torch.load(f"path//best_model_stain_{self.test_id}_{self.name}.pth"))
            self.model.to(self.device)
            self.ema_model.load_state_dict(torch.load(f"path//best_model_stain_{self.test_id}_ema_{self.name}.pth"))
            self.ema_model.to(self.device)
            self.calc_epoch("test")
            self.show_result("test")
        wandb.finish()

    def calc_epoch(self, mode):
        if mode == "train":
            self.model.train()
            loader = itertools.islice(self.train_loader, self.patches_per_epoch)
            patches_num = self.patches_per_epoch
            grad_ctx = torch.enable_grad()

            if self.lr_epoch != 0:
                if self.epoch == self.lr_epoch:
                    # self.weight = 0
                    self.w_target_ssim = 5 * self.w_target_ssim
                    self.w_target_lpips = 5 * self.w_target_lpips

        elif mode == "val":
            self.model.eval()
            # loader = itertools.islice(self.val_loader, self.patches_per_epoch_val)
            # patches_num = self.patches_per_epoch_val
            loader = self.val_loader
            patches_num = len(loader)
            grad_ctx = torch.no_grad()
        elif mode == "test":
            self.model.eval()
            # loader = itertools.islice(self.test_loader, self.patches_per_epoch_val)
            # patches_num = self.patches_per_epoch_val
            loader = self.test_loader
            patches_num = len(loader)
            grad_ctx = torch.no_grad()
        else:
            raise NotImplementedError
        total_metrics_dict = None

        with grad_ctx:
            for ph1, ph2, real in tqdm(loader, total=patches_num):
                metrics = self.calc_batch(ph1, ph2, real, mode)
                # print(metrics)
                if total_metrics_dict is None:
                    total_metrics_dict = {k: 0 for k, v in metrics.items()}
                for k, v in metrics.items():
                    total_metrics_dict[k] += metrics[k].mean().item()

        for k, v in total_metrics_dict.items():
            total_metrics_dict[k] /= patches_num
        self.save_results(total_metrics_dict, mode)

    def calc_batch(self, ph1, ph2, real, mode):
        if mode == "train":
            ph1, ph2, real = self.vae_encode([ph1, ph2, real])
            return self.calc_matrix_train(ph1, ph2, real)
        else:
            metrics_dict = None
            len_met = 0
            _, _, H, W = ph1.shape
            n = self.patches_per_epoch_val
            for y_i in range(H // 2, H - self.img_size + 1, self.img_size):
                for x_i in range(W // 2, W - self.img_size + 1, self.img_size):
                    len_met += 1
                    # print(ph1[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size].shape)
                    ph1_vae, ph2_vae = self.vae_encode([
                        ph1[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size],
                        ph2[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size]
                    ])
                    real_vae = real[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size]
                    metrics = self.calc_matrix_test(ph1_vae, ph2_vae, real_vae)
                    if metrics_dict is None:
                        metrics_dict = {k: 0 for k, v in metrics.items()}
                    for k, v in metrics.items():
                        metrics_dict[k] += metrics[k]
                    n = n - 1
                    if n <= 0:
                        break
                if n <= 0:
                    break

            for k, v in metrics_dict.items():
                metrics_dict[k] /= len_met
            return metrics_dict
            # return self.calc_matrix_test(ph1, ph2, real)

    def vae_encode(self, images):
        images = images.to(self.device)
        with torch.no_grad():
            for i in range(len(images)):
                mu, log_var = self.vae.encode(images[i])
                # 確率的サンプリング または 単に mu を使う
                latents = self.vae.reparameterize(mu, log_var)
                # Stable Diffusionでは値をスケーリングすることが多い（分散を1に近づけるため）
                images[i] = latents * 0.18215
        return images

    def vae_decode(self, image):
        image = image.to(self.device)
        # 1. Latentをスケーリング戻し
        generated_latents = image / 0.18215
        # 2. VAEデコーダーで画像に戻す
        with torch.no_grad():
            return self.vae.decode(generated_latents)

    def calc_matrix_train(self, ph1, ph2, real):
        x = torch.concat([ph1, ph2], dim=1).to(self.device)
        real = real.to(self.device)
        # predict = F.sigmoid(self.model(x))

        t = self.diffusion.sample_timesteps(real.shape[0]).to(self.device)
        x_t, noise, sqrt_alpha_hat = self.diffusion.noise_images(real, t)
        if np.random.random() < self.no_label:
            label_train = True
            x = None
        else:
            label_train = False
        if self.real_pred:
            prediction = self.model(x_t, t, x)
            loss_mse = self.loss_fn_mse(noise, prediction[:, 0:1, :, :])
            if label_train:
                loss_ssim = torch.tensor(0.0)
                loss_target_ssim = torch.tensor(0.0)
                loss_target_lpips = torch.tensor(0.0)
            else:
                loss_ssim = self.loss_fn_ssim(real, prediction[:, 1:2, :, :])
                # loss_ssim = torch.tensor(0.0)
                output = 1 / sqrt_alpha_hat * (x_t - torch.sqrt(1 - sqrt_alpha_hat ** 2) * prediction[:, 0:1, :, :])
                # t_target_weight = 1 - 1 / (1 + torch.exp(-1 * (t.detach() - 500) / 50))
                t_target_weight = 1
                # plt.imshow(output[0][0].detach().cpu().numpy())
                # plt.axis("off")
                # plt.show()
                loss_target_ssim = (self.loss_fn_ssim(real, output) * t_target_weight)
                loss_target_lpips = (self.loss_fn_lpips(real, output))
                # print(t_target_weight)
                # print(loss_target)

            loss = (self.weight * loss_mse + self.w_ssim * loss_ssim + self.w_target_ssim * loss_target_ssim
                    + self.w_target_lpips * loss_target_lpips).mean()
            # print(loss)
            output_dict = {"mse": loss_mse, "ssim": loss_ssim, "target_ssim": loss_target_ssim,
                           "target_lpips": loss_target_lpips, "loss": loss}
        else:
            predicted_noise = self.model(x_t, t, x)
            loss = self.loss_fn_mse(noise, predicted_noise)
            output_dict = {"mse": loss}

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.ema.step_ema(self.ema_model, self.model)
        return output_dict

    def calc_matrix_test(self, ph1, ph2, real):
        x = torch.concat([ph1, ph2], dim=1).to(self.device)
        real = real.to(self.device)

        sampled_images = self.vae_decode(self.diffusion.sample(
            self.model, labels=x, num_inference_steps=self.num_inference_steps,
            noise_add=self.noise_add, cfg_scale=self.cfg_scale
        ))
        # print(real.shape, sampled_images.shape, sampled_images.shape)
        mse_loss = self.loss_fn_mse(real, sampled_images)
        ssim_loss = self.loss_fn_ssim(real, sampled_images)
        lpips_loss = self.loss_fn_lpips(real, sampled_images)

        ema_sampled_images = self.vae_decode(self.diffusion.sample(
            self.ema_model, labels=x, num_inference_steps=self.num_inference_steps,
            noise_add=self.noise_add, cfg_scale=self.cfg_scale
        ))
        mse_loss_ema = self.loss_fn_mse(real, ema_sampled_images)
        ssim_loss_ema = self.loss_fn_ssim(real, ema_sampled_images)
        lpips_loss_ema = self.loss_fn_lpips(real, ema_sampled_images)

        return {
            "mse": mse_loss,
            "mse_ema": mse_loss_ema,
            "ssim": ssim_loss,
            "ssim_ema": ssim_loss_ema,
            "lpips": lpips_loss,
            "lpips_ema": lpips_loss_ema,
        }

    def save_results(self, total_metrics_dict, mode):
        self.hist[mode].append(total_metrics_dict)
        for k, v in total_metrics_dict.items():
            print(f"{mode} {k}: {v}")
        if mode == "test":
            if self.test_id == "final":
                total_metrics_dict_data = [[k, v] for k, v in total_metrics_dict.items()]
                wandb.log({f"{mode}/{self.test_id}": wandb.Table(data=total_metrics_dict_data, columns=["index", "score"])})
            else:
                self.run.log({f"{mode}/{self.test_id}": total_metrics_dict[self.test_id],
                              "ema": 0})
                self.run.log({f"{mode}/{self.test_id}": total_metrics_dict[f"{self.test_id}_ema"],
                              "ema": 1})
            if self.test_id == "lpips":
                self.run.summary["lpips"] = total_metrics_dict["lpips"]
                self.run.summary["lpips_ema"] = total_metrics_dict["lpips_ema"]
        else:
            total_metrics_dict_log = {f"{mode}/" + k: v for k, v in total_metrics_dict.items()}
            epoch_dict = {"epoch": self.epoch}
            self.run.log(dict(**total_metrics_dict_log, **epoch_dict))

        if mode == "val":
            torch.save(self.model.state_dict(), f"path//best_model_stain_final_{self.name}.pth")
            torch.save(self.ema_model.state_dict(), f"path//best_model_stain_final_ema_{self.name}.pth")
            for val_n in range(len(self.val_list)):
                mean_loss = total_metrics_dict[self.val_list[val_n]]
                if mean_loss < self.best_score_list[val_n]:
                    print(f"Loss_{self.val_list[val_n]} improved to {mean_loss}, saving model")
                    self.best_score_list[val_n] = mean_loss
                    self.min_epoch_list[val_n] = self.epoch
                    if val_n < len(self.val_list) / 2:
                        torch.save(self.model.state_dict(), f"path//best_model_stain_{self.val_list[val_n]}_{self.name}.pth")
                    else:
                        torch.save(self.ema_model.state_dict(), f"path//best_model_stain_{self.val_list[val_n]}_{self.name}.pth")

    def show_result(self, mode, image_n=1):
        if mode == "val":
            loader = self.val_loader
            # n = 1 + image_n
            # n_lim = image_n
            n = 1
        elif mode == "test":
            loader = self.test_loader
            n = 4
            # n_lim = 0
        else:
            loader = self.train_loader
            n = 1
            # n_lim = 0
        n_x = 0
        for ph1, ph2, real in loader:
            # if n_x >= n_lim:
            output_pred, output_pred_ema, real = self.make_prediction(mode, ph1, ph2, real)
            for i in range(len(output_pred)):
                # if n_x >= n_lim:
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(real[i])
                axs[0].axis('off')
                axs[0].set_title('target')
                axs[1].imshow(output_pred[i])
                axs[1].axis('off')
                axs[1].set_title('prediction')
                axs[2].imshow(output_pred_ema[i])
                axs[2].axis('off')
                axs[2].set_title('ema_prediction')
                plt.tight_layout()
                plt.show()
                if mode == "val":
                    wandb.log({f"{mode}_image/normal": wandb.Image((output_pred[i] + 1) * 0.5 * 255.0),
                               f"{mode}_image/ema": wandb.Image((output_pred_ema[i] + 1) * 0.5 * 255.0),
                               "epoch": self.epoch})
                    if self.epoch == 0:
                        # real = real[:][0].cpu().detach().numpy()
                        wandb.log({f"{mode}_image/target": wandb.Image((real[i] + 1) * 0.5 * 255.0), })
                if mode == "test":
                    wandb.log({f"{mode}_image/{self.test_id}": wandb.Image((output_pred[i] + 1) * 0.5 * 255.0),
                               f"{mode}_image/ema_{self.test_id}": wandb.Image((output_pred_ema[i] + 1) * 0.5 * 255.0),
                               "n": n_x})
                    if self.test_id == "final":
                        # real = real[:][0].cpu().detach().numpy()
                        wandb.log({f"{mode}_image/target": wandb.Image((real[i] + 1) * 0.5 * 255.0),
                                   "n": n_x})
                n_x += 1
                if n_x >= n:
                    break
            if n_x >= n:
                break

    #     for ph1, ph2, real in loader:
    #         if n_x < n:
    #             output_pred, output_pred_ema, real = self.make_prediction(mode, ph1, ph2, real)
    #             for i in range(len(output_pred)):
    #                 if n_x < n:
    #                     self.show_images(real[i], output_pred[i], output_pred_ema[i], mode, n_x)
    #                     condition_n_x[c[i].item()] += 1

    def result_image(self, n_first=1, n_long=1, num_inference_steps=50, noise_add=True, cfg_scale=3, model="normal"):
        loader = self.val_loader
        n = n_first + n_long - 1
        for ph1, ph2, real in loader:
            if n <= n_long:
                _, _, H, W = ph1.shape
                x = torch.cat([ph1[0:1].to(self.device), ph2[0:1].to(self.device)], dim=1)
                with torch.no_grad():
                    if model == "normal":
                        pred = self.diffusion.sample(self.model, labels=x, num_inference_steps=num_inference_steps,
                                                     noise_add=noise_add, cfg_scale=cfg_scale)
                    else:
                        pred = self.diffusion.sample(self.ema_model, labels=x, num_inference_steps=num_inference_steps,
                                                     noise_add=noise_add, cfg_scale=cfg_scale)
                output_pred = pred[0][0].cpu().detach().numpy()
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].imshow(real[0].squeeze())
                axs[0].axis('off')
                axs[0].set_title('target')
                axs[1].imshow(output_pred)
                axs[1].axis('off')
                axs[1].set_title('prediction')
                plt.tight_layout()
                plt.show()
            n = n - 1
            if n <= 0:
                break

    def make_prediction(self, mode, ph1, ph2, real):
        if mode == "val":
            pred_size = self.img_size * self.val_crop
        else:
            pred_size = self.pred_size

        stride = self.img_size // 2
        _, _, H, W = ph1.shape
        # H_res = H % stride
        # W_res = W % stride
        # y_range = np.arange(0, H - self.img_size + 1, stride)
        # x_range = np.arange(0, W - self.img_size + 1, stride)
        # if H_res > 0:
        #     y_range = np.append(y_range, (H - self.img_size))
        # if W_res > 0:
        #     x_range = np.append(x_range, (W - self.img_size))
        H_range = [H // 2 - pred_size // 2, H // 2 + pred_size // 2]
        W_range = [W // 2 - pred_size // 2, W // 2 + pred_size // 2]

        ph1 = ph1[:, :, H_range[0]: H_range[1], W_range[0]: W_range[1]]
        ph2 = ph2[:, :, H_range[0]: H_range[1], W_range[0]: W_range[1]]
        real = real[:, :, H_range[0]: H_range[1], W_range[0]: W_range[1]]
        y_range = np.arange(0, pred_size - self.img_size + 1, stride)
        x_range = np.arange(0, pred_size - self.img_size + 1, stride)

        # 予測結果と重みを蓄積するためのTensorをGPU上に初期化
        predictions_sum = torch.zeros_like(ph1, dtype=torch.float32).to(self.device)
        predictions_sum_ema = torch.zeros_like(ph1, dtype=torch.float32).to(self.device)
        weights_sum = torch.zeros_like(ph1, dtype=torch.float32).to(self.device)

        # 重み付けマップを生成 (GPU上に作成)
        blending_mask = create_gaussian_blending_mask(self.img_size, self.device)

        # 推論中は勾配計算を無効化してメモリ効率を上げる
        with torch.no_grad():
            # y方向（縦）にスライド
            for y_i in tqdm(y_range):
                # x方向（横）にスライド
                for x_i in x_range:
                    # GPU上のTensorから直接パッチを切り出す
                    # patch = x[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size]
                    phases = self.vae_encode([
                        ph1[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size],
                        ph2[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size]
                    ])
                    patch = torch.concat([phases[0], phases[1]], dim=1).to(self.device)

                    # print(x.shape)
                    # print(patch.shape)

                    # モデルで予測を実行 (GPU上で計算)
                    predicted_patch = self.vae_decode(self.diffusion.sample(
                        self.model, labels=patch, num_inference_steps=self.num_inference_steps,
                        noise_add=self.noise_add, cfg_scale=self.cfg_scale
                    ))
                    predicted_patch_ema = self.vae_decode(self.diffusion.sample(
                        self.ema_model, labels=patch, num_inference_steps=self.num_inference_steps,
                        noise_add=self.noise_add, cfg_scale=self.cfg_scale
                    ))

                    # 予測結果と重みを対応する位置に加算 (GPU上で計算)
                    predictions_sum[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size] += predicted_patch * blending_mask
                    predictions_sum_ema[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size] += predicted_patch_ema * blending_mask
                    weights_sum[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size] += blending_mask

                    if not self.val_crop:
                        if mode == "val":
                            break
                if not self.val_crop:
                    if mode == "val":
                        break

        # ゼロ除算を避ける
        weights_sum[weights_sum == 0] = 1.0

        # 加重平均を計算して最終的な予測結果を得る (GPU上で計算)
        fake = (predictions_sum / weights_sum)[:, 0, :, :].cpu().detach().numpy()
        fake_ema = (predictions_sum_ema / weights_sum)[:, 0, :, :].cpu().detach().numpy()
        real = real[:, 0, :, :].cpu().detach().numpy()

        # else:
        #     with torch.no_grad():
        #         pred = self.diffusion.sample(self.model, n=1, labels=x,
        #                                                 num_inference_steps=self.num_inference_steps,
        #                                                 noise_add=self.noise_add, cfg_scale=self.cfg_scale)
        #         pred_ema = self.diffusion.sample(self.ema_model, n=1, labels=x,
        #                                                     num_inference_steps=self.num_inference_steps,
        #                                                     noise_add=self.noise_add, cfg_scale=self.cfg_scale)
        #     output_pred = pred[0][0]
        #     fake = F.sigmoid(output_pred).cpu().detach().numpy()
        #     output_pred = pred_ema[0][0]
        #     fake_ema = F.sigmoid(output_pred).cpu().detach().numpy()

        return fake, fake_ema, real