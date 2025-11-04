import numpy as np
import copy
import itertools
import os

import torch
import lpips
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from pytorch_msssim import ssim
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import asdict

import wandb
from config import DigitalStainingConfig
from make_dataset import DatasetDigitalStaining
from discriminator_modules import *
from generator_modules import *


# %%
def tensor_ssim(img1, img2):
    return 1.0 - ssim(img1, img2, data_range=1.0, size_average=True)

def dice_loss_calc(pred, target, smooth=1):
    """
    Computes the Dice Loss for binary segmentation.
    Args:
        pred: Tensor of predictions (batch_size, 1, H, W).
        target: Tensor of ground truth (batch_size, 1, H, W).
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Scalar Dice Loss.
    """
    # Apply sigmoid to convert logits to probabilities
    # pred = torch.sigmoid(pred)
    # Calculate intersection and union
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    # Compute Dice Coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    # Return Dice Loss
    return 1 - dice.mean()

def lpips_calc(img1, img2, loss_fn):
    loss = loss_fn(img1, img2)
    return loss.mean()

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


class StainingGAN():
    def __init__(self, config: DigitalStainingConfig):
        self.config = config

        self.dir = self.config.dir
        self.train_folders = self.config.train_folders
        self.val_folders = self.config.val_folders
        self.test_folders = self.config.test_folders
        self.name = self.config.name
        self.group = self.config.group
        self.target = self.config.target
        self.n_epoch = self.config.n_epoch
        self.discriminator = self.config.discriminator
        self.num_workers = self.config.num_workers
        self.in_chans = self.config.in_chans
        self.w_l1 = self.config.w_l1
        self.w_ssim = self.config.w_ssim
        self.w_dice = self.config.w_dice
        self.img_size = self.config.img_size
        self.lr_g = self.config.learning_rate_g  # Note: variable name shortened
        self.lr_d = self.config.learning_rate_d  # Note: variable name shortened
        self.betas = self.config.betas
        self.images_to_use = self.config.images_to_use
        self.device = self.config.device
        self.patches_per_epoch = self.config.patches_per_epoch
        self.patches_per_epoch_val = self.config.patches_per_epoch_val
        self.val_epoch = self.config.val_epoch
        self.batch_size = self.config.batch_size
        self.decoder_attention_type = self.config.decoder_attention_type
        self.val_crop = self.config.val_crop
        self.epoch_start_ema = self.config.epoch_start_ema
        self.dim_mults = self.config.dim_mults
        self.self_attension = self.config.self_attension
        self.generator = self.config.generator
        self.encoder_name = self.config.encoder_name
        self.ema_beta = self.config.ema_beta
        self.plt_show = self.config.plt_show

        self.step_start_ema = self.epoch_start_ema * self.patches_per_epoch
        self.target_class = len(self.target)
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.l1_loss = torch.nn.L1Loss().to(self.device)
        self.loss_fn_mse = torch.nn.MSELoss().to(self.device)
        self.loss_fn_ssim = tensor_ssim
        self.loss_fn_lpips = lpips.LPIPS(net='alex').to(self.device)
        self.w_adv = 0.0 if self.discriminator == "U_Net" else 1.0
        self.last_batch_with_pred = None
        self.class_num = 1
        self.hist = {"train": [], "val": [], "test": []}
        self.test_id = ""
        self.val_list = [
            "mse",
            "ssim",
            "lpips",
            "mse_ema",
            "ssim_ema",
            "lpips_ema",
        ]
        self.best_score_list = [float('inf')] * len(self.val_list)
        self.min_epoch_list = [0] * len(self.val_list)
        self.epoch = 0

        if self.generator == "conditional":
            self.G = UNet_gemini(
                c_in=2,  # ノイズ画像(3ch) + 条件画像(3ch)
                c_out=1,  # ノイズを予測
                base_dim=64,  # ベースとなるチャンネル数
                dim_mults=self.dim_mults,  # 解像度ごとのチャンネル数の倍率
                num_embeddings=self.target_class,
                self_attension=self.self_attension,
                device=self.device
            ).to(self.device)
        elif self.generator == "conditional_pretrained":
            self.G = UNet_gemini_pretrained_ori(
                c_in=2,
                c_out=1,
                num_embeddings=self.target_class,
                self_attension=self.self_attension,
                encoder_name=self.encoder_name,
                device=self.device
            ).to(self.device)
        else:
            self.G = smp.Unet(
                encoder_name=self.encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=self.in_chans,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=self.class_num,  # model output channels (number of classes in your dataset)
                decoder_attention_type=self.decoder_attention_type
            ).to(self.device)


        self.unet = True
        if self.discriminator == "Patch3":
            self.D = Patch(in_channels=self.in_chans + self.class_num, depth=3).to(self.device)
        elif self.discriminator == "Patch4":
            self.D = Patch(in_channels=self.in_chans + self.class_num, depth=4).to(self.device)
        elif self.discriminator == "Patch5":
            self.D = Patch(in_channels=self.in_chans + self.class_num, depth=5).to(self.device)
        elif self.discriminator == "ResnetPatch":
            self.D = ResnetPatch(in_channels=self.in_chans + self.class_num).to(self.device)
        elif self.discriminator == "Patch_projection":
            self.D = Patch_projection(in_channels=self.in_chans + self.class_num).to(self.device)
        elif self.discriminator == "Resnet":
            self.D = timm.create_model("resnet18",
                                       in_chans=self.in_chans + self.class_num,
                                       pretrained=False,
                                       num_classes=1).to(self.device)
        else:
            self.D = Patch(in_channels=self.in_chans + self.class_num, depth=1).to(self.device)
            self.unet = False
        self.optimizer_g = torch.optim.Adam(self.G.parameters(), lr=self.lr_g, betas=self.betas)
        self.optimizer_d = torch.optim.Adam(self.D.parameters(), lr=self.lr_d, betas=self.betas)
        self.ema = EMA(self.ema_beta)
        self.ema_G = copy.deepcopy(self.G).eval().requires_grad_(False)

        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.train_loader = self.make_loader(self.train_folders, "train")
        self.val_loader = self.make_loader(self.val_folders, "val")
        self.test_loader = self.make_loader(self.test_folders, "test")

    def make_loader(self, folders, mode):
        datasets = []
        for target_n in range(self.target_class):
            for f in folders[target_n]:
                img_folder = os.path.join(self.dir[target_n], f)
                datasets.append(DatasetDigitalStaining(img_folder, target_n, augmentation=None))
        combined_dataset = ConcatDataset(datasets)
        if mode == "train":
            return DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=True,
                              num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers>0)
        else:
            return DataLoader(combined_dataset, batch_size=4, shuffle=False,
                              num_workers=0, pin_memory=True)

    def _wandb_init(self):
        self.run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="kohei_tokyo-the-university-of-tokyo",
            # Set the wandb project where this run will be logged.
            project="Digital_Staining",
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
            self.G.load_state_dict(torch.load(f"path//best_model_G_stain_{self.test_id}_{self.name}.pth"))
            self.ema_G.load_state_dict(torch.load(f"path//best_model_G_stain_{self.test_id}_ema_{self.name}.pth"))
            self.G.to(self.device)
            self.ema_G.to(self.device)
            self.D.load_state_dict(torch.load(f"path//best_model_D_stain_{self.test_id}_{self.name}.pth"))
            self.D.to(self.device)
            self.calc_epoch("test")
            self.show_result("test")
        wandb.finish()

    def calc_epoch(self, mode):
        if mode == "train":
            self.G.train()
            if self.unet:
                self.D.train()
            else:
                self.D.eval()
            loader = itertools.islice(self.train_loader, self.patches_per_epoch)
            patches_num = self.patches_per_epoch
            grad_ctx = torch.enable_grad()
        elif mode == "val":
            self.G.eval()
            self.D.eval()
            loader = self.val_loader
            patches_num = len(loader)
            # loader = itertools.islice(self.val_loader, self.patches_per_epoch_val)
            # patches_num = self.patches_per_epoch_val
            grad_ctx = torch.no_grad()
        elif mode == "test":
            self.G.eval()
            self.D.eval()
            loader = self.test_loader
            patches_num = len(loader)
            # loader = itertools.islice(self.test_loader, self.patches_per_epoch_val)
            # patches_num = self.patches_per_epoch_val
            grad_ctx = torch.no_grad()
        else:
            raise NotImplementedError
        total_metrics_dict = None

        with grad_ctx:
            for ph1, ph2, real, c in tqdm(loader, total=patches_num):
                metrics = self.calc_batch(ph1, ph2, real, c, mode)
                if total_metrics_dict is None:
                    total_metrics_dict = {k: 0 for k, v in metrics.items()}
                for k, v in metrics.items():
                    total_metrics_dict[k] += metrics[k].item()

        for k, v in total_metrics_dict.items():
            total_metrics_dict[k] /= patches_num
        self.save_results(total_metrics_dict, mode)

    def calc_batch(self, ph1, ph2, real, c, mode):
        if mode == "train":
            return self.calc_matrix(ph1, ph2, real, c, mode)
        else:
            if self.val_crop:
                metrics_dict = None
                len_met = 0
                _, _, H, W = ph1.shape
                for y_i in range(0, H - self.img_size + 1, self.img_size):
                    for x_i in range(0, W - self.img_size + 1, self.img_size):
                        len_met += 1
                        metrics = self.calc_matrix(
                            ph1[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size],
                            ph2[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size],
                            real[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size],
                            c,
                            mode
                        )
                        if metrics_dict is None:
                            metrics_dict = {k: 0 for k, v in metrics.items()}
                        for k, v in metrics.items():
                            metrics_dict[k] += metrics[k]

                        if len_met >= self.patches_per_epoch_val:
                            break
                    if len_met >= self.patches_per_epoch_val:
                        break

                for k, v in metrics_dict.items():
                    metrics_dict[k] /= len_met
                return metrics_dict
            else:
                return self.calc_matrix(ph1, ph2, real, mode)

    def calc_matrix(self, ph1, ph2, real, c, mode):
        x = torch.concat([ph1, ph2], dim=1).to(self.device)
        c = c.to(self.device)
        real = real.to(self.device)
        # real_mask = real_mask.unsqueeze(1).to(self.device)
        # fake, fake_mask = self.G(x).split(1, dim=1)
        fake = F.sigmoid(self.G(x, c))
        fake_ema = F.sigmoid(self.ema_G(x, c))
        # real_pair = torch.cat([x, real, real_mask], dim=1)
        # fake_pair = torch.cat([x, fake, fake_mask], dim=1)
        real_pair = torch.cat([x, real], dim=1)
        fake_pair = torch.cat([x, fake], dim=1)

        loss_dis = self.calc_dis(real_pair, fake_pair, c, mode)
        # adv_loss, l1_loss, loss_g, mse, ssim_loss, dice_loss = self.calc_gen(
        #     real_pair, fake_pair, mode, real, real_mask, fake, fake_mask
        # )
        loss_gen = self.calc_gen(
            real_pair, fake_pair, c, mode, real, fake, fake_ema
        )

        return dict(**loss_dis, **loss_gen)

    def calc_dis(self, real_pair, fake_pair, c, mode):
        pred_real = self.D(real_pair, c)
        pred_fake = self.D(fake_pair.detach(), c)
        target_real = torch.ones_like(pred_real).to(self.device)
        target_fake = torch.zeros_like(pred_fake).to(self.device)

        loss_fake = self.bce_loss(pred_fake, target_fake)
        loss_real = self.bce_loss(pred_real, target_real)
        loss_d = (loss_real + loss_fake) * 0.5
        acc_real = pred_real.sigmoid().float().mean()
        acc_fake = pred_fake.sigmoid().float().mean()
        loss_dict = {"loss_d": loss_d, "acc_real": acc_real, "acc_fake": acc_fake}

        if mode == "train":
            if self.unet:
                self.optimizer_d.zero_grad()
                loss_d.backward()
                self.optimizer_d.step()
        return loss_dict

    def calc_gen(self, real_pair, fake_pair, c, mode, real, fake, fake_ema):
        pred_fake = self.D(fake_pair, c)
        target_fake = torch.ones_like(pred_fake).to(self.device)

        adv_loss = self.bce_loss(pred_fake, target_fake)
        l1_loss = self.l1_loss(real, fake)
        ssim_loss = self.loss_fn_ssim(real, fake)
        mse = self.loss_fn_mse(real, fake)
        if mode == "train":
            lpips_loss = torch.tensor(0.0)
        # dice_loss = dice_loss_calc(fake_mask, real_mask)
        # loss_g = adv_loss + self.w_l1 * l1_loss + self.w_ssim * ssim_loss + self.w_dice * dice_loss
        loss_g = self.w_adv * adv_loss + self.w_l1 * l1_loss + self.w_ssim * ssim_loss

        if mode == "train":
            self.optimizer_g.zero_grad()
            loss_g.backward()
            self.optimizer_g.step()
            self.ema.step_ema(self.ema_G, self.G, step_start_ema=self.step_start_ema)
            loss_dict = {
                "adv_loss_g": adv_loss, "l1_loss": l1_loss, "loss_g": loss_g, "mse": mse, "ssim": ssim_loss
            }
        else:
            lpips_loss = lpips_calc(real, fake, self.loss_fn_lpips)
            l1_loss_ema = self.l1_loss(real, fake_ema)
            ssim_loss_ema = self.loss_fn_ssim(real, fake_ema)
            mse_ema = self.loss_fn_mse(real, fake_ema)
            lpips_ema = lpips_calc(real, fake_ema, self.loss_fn_lpips)
            loss_dict = {
                "adv_loss_g": adv_loss, "l1_loss": l1_loss, "loss_g": loss_g, "mse": mse, "ssim": ssim_loss,
                "lpips": lpips_loss, "l1_loss_ema": l1_loss_ema, "ssim_ema": ssim_loss_ema, "mse_ema": mse_ema,
                "lpips_ema": lpips_ema,
            }
        return loss_dict

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
            torch.save(self.G.state_dict(), f"path//best_model_G_stain_final_{self.name}.pth")
            torch.save(self.ema_G.state_dict(), f"path//best_model_G_stain_final_ema_{self.name}.pth")
            torch.save(self.D.state_dict(), f"path//best_model_D_stain_final_{self.name}.pth")
            for val_n in range(len(self.val_list)):
                mean_loss = total_metrics_dict[self.val_list[val_n]]
                if mean_loss < self.best_score_list[val_n]:
                    print(f"Loss_{self.val_list[val_n]} improved to {mean_loss}, saving model")
                    self.best_score_list[val_n] = mean_loss
                    self.min_epoch_list[val_n] = self.epoch
                    if val_n < len(self.val_list) / 2:
                        torch.save(self.G.state_dict(), f"path//best_model_G_stain_{self.val_list[val_n]}_{self.name}.pth")
                        torch.save(self.D.state_dict(), f"path//best_model_D_stain_{self.val_list[val_n]}_{self.name}.pth")
                    else:
                        torch.save(self.ema_G.state_dict(), f"path//best_model_G_stain_{self.val_list[val_n]}_{self.name}.pth")

    def show_result(self, mode):
        if mode == "val":
            loader = self.val_loader
            n = 1
        elif mode == "test":
            loader = self.test_loader
            n = 4
        else:
            loader = self.train_loader
            n = 1

        condition_n_x = [0] * self.target_class
        for ph1, ph2, real, c in loader:
            if condition_n_x[c[0].item()] < n:
                fake, fake_ema, real = self.make_fake(ph1, ph2, real, c)
                for i in range(len(fake)):
                    if condition_n_x[c[i].item()] < n:
                        self.show_images(real[i], fake[i], fake_ema[i], c[i].item(), mode, condition_n_x)
                        condition_n_x[c[i].item()] += 1

    def make_fake(self, ph1, ph2, real, c):
        x = torch.cat([ph1.to(self.device), ph2.to(self.device)], dim=1)
        c = c.to(self.device)
        if self.val_crop:
            stride = self.img_size // 2
            _, _, H, W = x.shape
            H_res = H % stride
            W_res = W % stride
            y_range = np.arange(0, H - self.img_size + 1, stride)
            x_range = np.arange(0, W - self.img_size + 1, stride)
            if H_res > 0:
                y_range = np.append(y_range, (H - self.img_size))
            if W_res > 0:
                x_range = np.append(x_range, (W - self.img_size))

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
                        patch = x[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size]

                        # モデルで予測を実行 (GPU上で計算)
                        predicted_patch = F.sigmoid(self.G(patch, c))
                        predicted_patch_ema = F.sigmoid(self.ema_G(patch, c))

                        # 予測結果と重みを対応する位置に加算 (GPU上で計算)
                        predictions_sum[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size] += predicted_patch * blending_mask
                        predictions_sum_ema[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size] += predicted_patch_ema * blending_mask
                        weights_sum[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size] += blending_mask

            # ゼロ除算を避ける
            weights_sum[weights_sum == 0] = 1.0

            # 加重平均を計算して最終的な予測結果を得る (GPU上で計算)
            fake = (predictions_sum / weights_sum)[:][0].cpu().detach().numpy()
            fake_ema = (predictions_sum_ema / weights_sum)[:][0].cpu().detach().numpy()

        else:
            with torch.no_grad():
                pred = self.G(x, c)  # [1,2,img_size,img_size]
                pred_ema = self.ema_G(x, c)
            output_pred = pred[:][0]
            fake = F.sigmoid(output_pred).cpu().detach().numpy()
            output_pred = pred_ema[:][0]
            fake_ema = F.sigmoid(output_pred).cpu().detach().numpy()

        return fake, fake_ema, real[:][0]

    def show_images(self, real, fake, fake_ema, c, mode, condition_n_x):
        if self.plt_show:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(real)
            axs[0].axis('off')
            axs[0].set_title(f'{self.target[c]}_target')
            axs[1].imshow(fake)
            axs[1].axis('off')
            axs[1].set_title(f'{self.target[c]}_prediction')
            axs[2].imshow(fake_ema)
            axs[2].axis('off')
            axs[2].set_title(f'{self.target[c]}_ema_prediction')
            plt.tight_layout()
            plt.show()
        if mode == "val":
            wandb.log({f"{mode}_image/{self.target[c]}_normal": wandb.Image(fake * 255.0),
                       f"{mode}_image/{self.target[c]}_ema": wandb.Image(fake_ema * 255.0),
                       "epoch": self.epoch})
            if self.epoch == 0:
                real = real.cpu().detach().numpy()
                wandb.log({f"{mode}_image/{self.target[c]}_target": wandb.Image(real * 255.0), })
        if mode == "test":
            wandb.log({f"{mode}_image/{self.target[c]}_{self.test_id}": wandb.Image(fake * 255.0),
                       f"{mode}_image/ema_{self.test_id}": wandb.Image(fake_ema * 255.0),
                       "n": condition_n_x[c]})
            if self.test_id == "final":
                real = real.cpu().detach().numpy()
                wandb.log({f"{mode}_image/{self.target[c]}_target": wandb.Image(real * 255.0),
                           "n": condition_n_x[c]})
