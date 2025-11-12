import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from pytorch_msssim import ssim
from dataclasses import asdict
import timm
import torch.nn.functional as F
import lpips
import itertools
import wandb
from make_dataset import DatasetDigitalStaining

def tensor_ssim(img1, img2):
    return 1.0 - ssim(img1, img2, data_range=1.0, size_average=True)

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

def lpips_calc(img1, img2, loss_fn):
    loss = loss_fn(img1, img2)
    return loss.mean()

# def lpips_calc_each(img1, img2, prompts, targets, loss_fn, name):
#     loss = loss_fn(img1, img2)
#     loss_each = {}
#     target_n = {}
#     loss_each[name] = loss.mean()
#     for target in targets:
#         loss_target = sum([loss[i] * (prompts[i] == target) for i in range(len(prompts))])
#         loss_each[f"{name}_{target}"] = loss_target
#         target_n[f"{name}_{target}"] = (sum([(prompts[i] == target) for i in range(len(prompts))]))
#     return loss_each, target_n

def lpips_calc_each(img1, img2, prompts, targets, loss_fn, name):
    """
    LPIPSなどのバッチごとの損失を計算し、プロンプト(カテゴリ)別に
    「損失の合計」と「サンプル数」を計算します。

    前提: loss_fn (例: lpips.LPIPS) は、
    reduction='none' で初期化されている必要があります。
    """

    # 1. バッチ内の画像ごとの損失を計算 [B, 1, 1, 1]
    loss_per_image = loss_fn(img1, img2)

    # 2. 形状を [B] (バッチサイズ) の1Dテンソルに平坦化
    loss_flat = loss_per_image.view(-1)

    # 3. 結果を保存する辞書
    loss_each = {}
    n_target = {}

    # 4. バッチ全体の平均損失 (元のコードの 'loss_each[name]' に対応)
    loss_each[name] = loss_flat.mean().item()

    # 5. プロンプト(文字列)のリストをNumpy配列に変換
    prompts_np = np.array(prompts)

    # 6. 各ターゲット（カテゴリ）ごとにループ
    for target in targets:
        key = f"{name}_{target}"

        # 7. ブール値マスクを作成 (例: prompts_np == "mito")
        mask_np = (prompts_np == target)

        # 8. このターゲットのサンプル数を計算
        count = mask_np.sum()
        n_target[key] = count  # サンプル数を辞書に格納

        if count > 0:
            # 9. マスクを使って該当する損失のみを抽出し、合計を計算
            mask_torch = torch.from_numpy(mask_np).to(loss_flat.device)
            loss_sum = loss_flat[mask_torch].sum().item()
            loss_each[key] = loss_sum  # 損失の合計を辞書に格納
        else:
            # 10. 該当するサンプルがバッチ内にない場合は 0.0
            loss_each[key] = 0.0

    return loss_each, n_target

class Result(nn.Module):
    def __init__(
            self,
            model_sample,
            model_sample_ema,
            train_loader,
            val_loader,
            test_loader,
            img_size,
            target=None,
            val_crop=1,
            min_max=[-1, 1],
            device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.model_sample = model_sample
        self.model_sample_ema = model_sample_ema
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.img_size = img_size
        self.target = target
        self.val_crop = val_crop
        self.min_max = min_max
        self.device = device


    def show_result(self, mode, epoch=0, test_id=None):
        self.epoch = epoch
        self.test_id = test_id
        if mode == "val":
            loader = self.val_loader
            n = 1
        elif mode == "test":
            loader = self.test_loader
            n = 4
        else:
            loader = self.train_loader
            n = 1

        # if self.target is None:
        #     n_x = 0
        #     for ph1, ph2, real in loader:
        #         if n_x < n:
        #             prediction, prediction_ema, real = self.make_prediction(mode, ph1, ph2, real)
        #             for i in range(len(prediction)):
        #                 if n_x < n:
        #                     self.show_images(mode, real[i], prediction[i], prediction_ema[i], n_x)
        #                     n_x += 1
        # else:
        condition_n_x = {prompt : 0 for prompt in self.target}
        for batch in loader:
            if condition_n_x[batch["prompt"][0]] < n:
                prediction, prediction_ema, target_images = self.make_prediction(mode, batch)
                for i in range(len(prediction)):
                    if condition_n_x[batch["prompt"][i]] < n:
                        self.show_images(mode, target_images[i][0], prediction[i][0], prediction_ema[i][0], condition_n_x, batch["prompt"][i])
                        condition_n_x[batch["prompt"][i]] += 1

    def make_prediction(self, mode, batch):
        target_images = batch["target_image"].to(self.device)
        conditioning_images = batch["conditioning_image"].to(self.device)
        prompt = batch["prompt"]
        stride = self.img_size // 2
        _, _, H, W = target_images.shape

        if (self.val_crop > 0) and (mode == "val"):
            pred_size = self.img_size * self.val_crop
            H_range = [H // 2 - pred_size // 2, H // 2 + pred_size // 2]
            W_range = [W // 2 - pred_size // 2, W // 2 + pred_size // 2]
            target_images = target_images[:, :, H_range[0]: H_range[1], W_range[0]: W_range[1]]
            conditioning_images = conditioning_images[:, :, H_range[0]: H_range[1], W_range[0]: W_range[1]]
            y_range = np.arange(0, pred_size - self.img_size + 1, stride)
            x_range = np.arange(0, pred_size - self.img_size + 1, stride)

        else:
            stride = self.img_size // 2
            H_res = H % stride
            W_res = W % stride
            y_range = np.arange(0, H - self.img_size + 1, stride)
            x_range = np.arange(0, W - self.img_size + 1, stride)
            if H_res > 0:
                y_range = np.append(y_range, (H - self.img_size))
            if W_res > 0:
                x_range = np.append(x_range, (W - self.img_size))

        # if c is not None:
        #     c = c.to(self.device)
        # 予測結果と重みを蓄積するためのTensorをGPU上に初期化
        predictions_sum = torch.zeros_like(target_images, dtype=torch.float32).to(self.device)
        predictions_sum_ema = torch.zeros_like(target_images, dtype=torch.float32).to(self.device)
        weights_sum = torch.zeros_like(target_images, dtype=torch.float32).to(self.device)

        # 重み付けマップを生成 (GPU上に作成)
        blending_mask = create_gaussian_blending_mask(self.img_size, self.device)

        # 推論中は勾配計算を無効化してメモリ効率を上げる
        with torch.no_grad():
            # y方向（縦）にスライド
            for y_i in tqdm(y_range):
                # x方向（横）にスライド
                for x_i in x_range:
                    # GPU上のTensorから直接パッチを切り出す
                    patch = conditioning_images[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size]

                    # モデルで予測を実行 (GPU上で計算)
                    # if c is None:
                    #     predicted_patch = self.model_sample(patch)
                    #     predicted_patch_ema = self.model_sample_ema(patch)
                    # else:
                    predicted_patch = self.model_sample(patch, prompt)
                    predicted_patch_ema = self.model_sample_ema(patch, prompt)

                    # 予測結果と重みを対応する位置に加算 (GPU上で計算)
                    predictions_sum[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size] += predicted_patch * blending_mask
                    predictions_sum_ema[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size] += predicted_patch_ema * blending_mask
                    weights_sum[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size] += blending_mask

        # ゼロ除算を避ける
        weights_sum[weights_sum == 0] = 1.0

        # 加重平均を計算して最終的な予測結果を得る (GPU上で計算)
        prediction = (predictions_sum / weights_sum).cpu().detach().numpy()
        prediction_ema = (predictions_sum_ema / weights_sum).cpu().detach().numpy()

        return prediction, prediction_ema, target_images.cpu().detach().numpy()

    def show_images(self, mode, target_images, prediction, prediction_ema, condition_n_x, prompt):
        min = self.min_max[0]
        max = self.min_max[1]
        target_images = (np.clip(target_images, min, max) + 1) * 255.0 / (max - min)
        prediction = (np.clip(prediction, min, max) + 1) * 255.0 / (max - min)
        prediction_ema = (np.clip(prediction_ema, min, max) + 1) * 255.0 / (max - min)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(target_images)
        axs[0].axis('off')
        axs[0].set_title(f'{prompt}_target' if prompt is not None else f'target')
        axs[1].imshow(prediction)
        axs[1].axis('off')
        axs[1].set_title(f'{prompt}_{self.test_id}_prediction' if prompt is not None else f'prediction')
        axs[2].imshow(prediction_ema)
        axs[2].axis('off')
        axs[2].set_title(f'{prompt}_ema_{self.test_id}_prediction' if prompt is not None else f'prediction_ema')
        plt.tight_layout()
        plt.show()
        if mode == "val":
            wandb.log({f"{mode}_image/{prompt}_normal" if prompt is not None else f"{mode}_image/normal": wandb.Image(prediction),
                       f"{mode}_image/{prompt}_ema" if prompt is not None else f"{mode}_image/ema": wandb.Image(prediction_ema),
                       "epoch": self.epoch})
            if self.epoch == 0:
                wandb.log({f"{mode}_image/{prompt}_target" if prompt is not None else f"{mode}_image/target": wandb.Image(target_images)})
        if mode == "test":
            wandb.log({f"{mode}_image/{prompt}_{self.test_id}" if prompt is not None else f"{mode}_image/{self.test_id}": wandb.Image(prediction),
                       f"{mode}_image/{prompt}_ema_{self.test_id}" if prompt is not None else f"{mode}_image/ema_{self.test_id}": wandb.Image(prediction_ema),
                       "n": condition_n_x[prompt] if prompt is not None else condition_n_x})
            if self.test_id == "final":
                wandb.log({f"{mode}_image/{prompt}_target" if prompt is not None else f"{mode}_image/target": wandb.Image(target_images),
                           "n": condition_n_x[prompt] if prompt is not None else condition_n_x})