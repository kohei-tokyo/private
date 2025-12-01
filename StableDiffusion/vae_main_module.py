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
import albumentations as A
import cv2
import torch.nn.functional as F
import lpips
import itertools
import wandb
from make_dataset import DatasetDigitalStaining
from ddpm_conditional import Diffusion_ddim, Diffusion_ddpm
from config import VAEConfig
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

class VAE(nn.Module):
    def __init__(self, config: VAEConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        # --- パスとデータに関する設定 ---
        self.dir = self.config.dir
        self.train_folders = self.config.train_folders
        self.val_folders = self.config.val_folders
        self.test_folders = self.config.test_folders
        self.images_to_use = self.config.images_to_use
        self.specialize = self.config.specialize

        # --- 実験管理 (W&Bなど) ---
        self.name = self.config.name
        self.group = self.config.group

        # --- 学習ループに関する設定 ---
        self.n_epoch = self.config.n_epoch
        self.learning_rate = self.config.learning_rate
        self.batch_size = self.config.batch_size
        self.patches_per_epoch = self.config.patches_per_epoch
        self.val_epoch = self.config.val_epoch
        self.w_kld = self.config.w_kld
        self.img_size = self.config.img_size

        # --- モデルのアーキテクチャ設定 ---
        self.latent_dim = self.config.latent_dim
        self.hidden_dims = self.config.hidden_dims

        # --- 環境設定 ---
        self.num_workers = self.config.num_workers
        self.device = self.config.device

        self.model = VAE_model(in_channels=1, latent_dim=self.latent_dim).to(self.device)  # Mito画像(1ch)を4chの潜在変数にする例
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.ema = EMA(0.9999, step_start_ema=self.n_epoch * self.patches_per_epoch / 2)
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
        datasets = [DatasetDigitalStaining(img_folders[i], augmentation=augment, specialize=self.specialize) for i in range(len(self.train_folders))]
        combined_dataset = ConcatDataset(datasets)
        self.batch_size_train = self.batch_size
        self.batch_size = 1
        self.train_loader = DataLoader(combined_dataset, batch_size=self.batch_size_train, shuffle=True,
                                       num_workers=self.num_workers,
                                       pin_memory=True, persistent_workers=self.num_workers > 0)

        # Validation DataLoader
        img_folders = [os.path.join(self.dir, f) for f in self.val_folders]
        datasets = [DatasetDigitalStaining(img_folders[i], augmentation=None, specialize=self.specialize) for i in range(len(self.val_folders))]
        combined_dataset = ConcatDataset(datasets)
        self.val_loader = DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=False,
                                     num_workers=self.num_workers, drop_last=True,
                                     pin_memory=True, persistent_workers=self.num_workers > 0)

        # Test DataLoader
        img_folders = [os.path.join(self.dir, f) for f in self.test_folders]
        datasets = [DatasetDigitalStaining(img_folders[i], augmentation=None, specialize=self.specialize) for i in range(len(self.test_folders))]
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

    def train(self):
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

    def test(self):
        test_list = ["final", "ssim", "lpips"]
        for self.test_id in test_list:
            print(f"Test {self.test_id}")
            self.model.load_state_dict(torch.load(f"path//best_model_vae_{self.test_id}_{self.name}.pth"))
            self.model.to(self.device)
            self.ema_model.load_state_dict(torch.load(f"path//best_model_vae_{self.test_id}_ema_{self.name}.pth"))
            self.ema_model.to(self.device)
            self.calc_epoch("test")
            self.show_result("test")
        wandb.finish()

    def calc_epoch(self, mode):
        if mode == "train":
            self.model.train()
            if self.patches_per_epoch == 0:
                loader = self.train_loader
                patches_num = len(loader)
            else:
                loader = itertools.islice(self.train_loader, self.patches_per_epoch)
                patches_num = self.patches_per_epoch
            grad_ctx = torch.enable_grad()
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
            for images in tqdm(loader, total=patches_num):
                metrics = self.calc_batch(images, mode)
                # print(metrics)
                if total_metrics_dict is None:
                    total_metrics_dict = {k: 0 for k, v in metrics.items()}
                for k, v in metrics.items():
                    total_metrics_dict[k] += metrics[k].mean().item()

        for k, v in total_metrics_dict.items():
            total_metrics_dict[k] /= patches_num
        self.save_results(total_metrics_dict, mode)

    def calc_batch(self, images, mode):
        if self.specialize is None:
            ph1, ph2, real = images
            images = [ph1, ph2, real]
        else:
            images = [images]
        if mode == "train":
            return self.calc_matrix(images, mode)
        else:
            metrics_dict = None
            len_met = 0
            _, _, H, W = images[0].shape
            # n = self.patches_per_epoch_val
            for y_i in range(H // 2, H - self.img_size + 1, self.img_size):
                for x_i in range(W // 2, W - self.img_size + 1, self.img_size):
                    len_met += 1
                    # print(ph1[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size].shape)
                    metrics = self.calc_matrix(
                        [image[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size] for image in images], mode
                    )
                    if metrics_dict is None:
                        metrics_dict = {k: 0 for k, v in metrics.items()}
                    for k, v in metrics.items():
                        metrics_dict[k] += metrics[k]
                #     n = n - 1
                #     if n <= 0:
                #         break
                # if n <= 0:
                #     break

            for k, v in metrics_dict.items():
                metrics_dict[k] /= len_met
            return metrics_dict

    def calc_matrix(self, images, mode):
        input_images = torch.cat(images, dim=0).to(self.device)

        # Forward
        recon_batch, input, mu, log_var = self.model(input_images)

        # Loss計算 (再構成誤差 + KLダイバージェンス)
        loss_recon = F.mse_loss(recon_batch, input)
        loss_kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = loss_recon + self.w_kld * loss_kld  # KLDの重みは小さく

        if mode == "train":
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.ema.step_ema(self.ema_model, self.model)
            output_dict = {"recon": loss_recon, "kld": loss_kld, "loss": loss}
        else:
            loss_ssim = self.loss_fn_ssim(recon_batch, input)
            loss_lpips = self.loss_fn_lpips(recon_batch, input)
            output_dict = {"recon": loss_recon, "kld": loss_kld, "ssim": loss_ssim, "lpips": loss_lpips, "loss": loss}

            # Forward
            recon_batch, input, mu, log_var = self.ema_model(input_images)
            # Loss計算 (再構成誤差 + KLダイバージェンス)
            loss_recon = F.mse_loss(recon_batch, input)
            loss_kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = loss_recon + self.w_kld * loss_kld  # KLDの重みは小さく
            loss_ssim = self.loss_fn_ssim(recon_batch, input)
            loss_lpips = self.loss_fn_lpips(recon_batch, input)
            output_dict_ema = {"recon_ema": loss_recon, "kld_ema": loss_kld, "ssim_ema": loss_ssim,
                           "lpips_ema": loss_lpips, "loss_ema": loss}
            output_dict = {**output_dict, **output_dict_ema}
        return output_dict

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
            torch.save(self.model.state_dict(), f"path//best_model_vae_final_{self.name}.pth")
            torch.save(self.ema_model.state_dict(), f"path//best_model_vae_final_ema_{self.name}.pth")
            for val_n in range(len(self.val_list)):
                mean_loss = total_metrics_dict[self.val_list[val_n]]
                if mean_loss < self.best_score_list[val_n]:
                    print(f"Loss_{self.val_list[val_n]} improved to {mean_loss}, saving model")
                    self.best_score_list[val_n] = mean_loss
                    self.min_epoch_list[val_n] = self.epoch
                    if val_n < len(self.val_list) / 2:
                        torch.save(self.model.state_dict(), f"path//best_model_vae_{self.val_list[val_n]}_{self.name}.pth")
                    else:
                        torch.save(self.ema_model.state_dict(), f"path//best_model_vae_{self.val_list[val_n]}_{self.name}.pth")

    def show_result(self, mode):
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
        for images in loader:
            if self.specialize is None:
                ph1, ph2, real = images
                images = [ph1, ph2, real]
            else:
                images = [images]
            # if n_x >= n_lim:
            output_pred, output_pred_ema, target = self.make_prediction(mode, images)
            for i_n in range(len(images[0])):
                for j in range(len(images)):
                    # if n_x >= n_lim:
                    i = i_n + len(images[0]) * j
                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                    axs[0].imshow(target[i])
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
                        wandb.log({f"{mode}_image/normal_{j}": wandb.Image((output_pred[i] + 1) * 0.5 * 255.0),
                                   f"{mode}_image/ema_{j}": wandb.Image((output_pred_ema[i] + 1) * 0.5 * 255.0),
                                   "epoch": self.epoch})
                        if self.epoch == 0:
                            # real = real[:][0].cpu().detach().numpy()
                            wandb.log({f"{mode}_image/target_{j}": wandb.Image((target[i] + 1) * 0.5 * 255.0), })
                    if mode == "test":
                        wandb.log({f"{mode}_image/{self.test_id}_{j}": wandb.Image((output_pred[i] + 1) * 0.5 * 255.0),
                                   f"{mode}_image/ema_{self.test_id}_{j}": wandb.Image((output_pred_ema[i] + 1) * 0.5 * 255.0),
                                   "n": n_x})
                        if self.test_id == "final":
                            # real = real[:][0].cpu().detach().numpy()
                            wandb.log({f"{mode}_image/target_{j}": wandb.Image((target[i] + 1) * 0.5 * 255.0),
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
                        pred = self.diffusion.sample(self.model, n=self.batch_size, labels=x, num_inference_steps=num_inference_steps,
                                                     noise_add=noise_add, cfg_scale=cfg_scale)
                    else:
                        pred = self.diffusion.sample(self.ema_model, n=self.batch_size, labels=x, num_inference_steps=num_inference_steps,
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

    def make_prediction(self, mode, images):
        stride = self.img_size // 2
        _, _, H, W = images[0].shape
        pred_size = min([H, W])
        # H_res = H % stride
        # W_res = W % stride
        # y_range = np.arange(0, H - self.img_size + 1, stride)
        # x_range = np.arange(0, W - self.img_size + 1, stride)
        # if H_res > 0:
        #     y_range = np.append(y_range, (H - self.img_size))
        # if W_res > 0:
        #     x_range = np.append(x_range, (W - self.img_size))
        H_range = [0, pred_size]
        W_range = [0, pred_size]

        images = [image[:, :, H_range[0]: H_range[1], W_range[0]: W_range[1]] for image in images]
        # ph1 = ph1[:, :, H_range[0]: H_range[1], W_range[0]: W_range[1]]
        # ph2 = ph2[:, :, H_range[0]: H_range[1], W_range[0]: W_range[1]]
        # real = real[:, :, H_range[0]: H_range[1], W_range[0]: W_range[1]]
        y_range = np.arange(0, pred_size - self.img_size + 1, stride)
        x_range = np.arange(0, pred_size - self.img_size + 1, stride)

        # 予測結果と重みを蓄積するためのTensorをGPU上に初期化
        predictions_sum = torch.zeros((images[0].shape[0] * 3, images[0].shape[1], images[0].shape[2], images[0].shape[3]),
                                      dtype=torch.float32).to(self.device)
        predictions_sum_ema = torch.zeros((images[0].shape[0] * 3, images[0].shape[1], images[0].shape[2], images[0].shape[3]),
                                          dtype=torch.float32).to(self.device)
        weights_sum = torch.zeros((images[0].shape[0] * 3, images[0].shape[1], images[0].shape[2], images[0].shape[3]),
                                  dtype=torch.float32).to(self.device)

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
                    patch = torch.cat([image[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size] for image in images],
                                      dim=0).to(self.device)

                    # print(x.shape)
                    # print(patch.shape)

                    # モデルで予測を実行 (GPU上で計算)
                    predicted_patch, _, _, _ = self.model(patch)
                    predicted_patch_ema, _, _, _ = self.ema_model(predicted_patch)

                    # 予測結果と重みを対応する位置に加算 (GPU上で計算)
                    predictions_sum[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size] += predicted_patch * blending_mask
                    predictions_sum_ema[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size] += predicted_patch_ema * blending_mask
                    weights_sum[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size] += blending_mask

                #     if not self.val_crop:
                #         if mode == "val":
                #             break
                # if not self.val_crop:
                #     if mode == "val":
                #         break

        # ゼロ除算を避ける
        weights_sum[weights_sum == 0] = 1.0

        # 加重平均を計算して最終的な予測結果を得る (GPU上で計算)
        fake = (predictions_sum / weights_sum)[:, 0, :, :].cpu().detach().numpy()
        fake_ema = (predictions_sum_ema / weights_sum)[:, 0, :, :].cpu().detach().numpy()
        target = torch.cat(images, dim=0)[:, 0, :, :].cpu().detach().numpy()
        return fake, fake_ema, target