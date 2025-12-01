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

from config import DDPMConfig
from make_dataset import DatasetDigitalStaining
from ddpm_conditional import Diffusion_ddim, Diffusion_ddpm
from utils import *
from modules import UNet_conditional, UNet_conditional_ori, UNet_conditional_deep, UNet_conditional_sa_5, UNet_conditional_dc_5, EMA
from modules_2 import UNet_gemini

def predict(
        img_folder,
        name="Test",
        test_id="lpips",
        img_size=128,
        n_result=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dim_mults=(1, 2, 4, 4, 8),
        mode_dif="ddim",
        noise_steps=1000,
        noise_add=True,
        cfg_scale=1,
        num_inference_steps=50,
        pred_size=512,
        target_steps=None,
        stop_steps=None,
):
    in_chans = 2
    class_num = 1
    model = UNet_gemini(
                c_in=in_chans + 1,
                c_out=2,
                dim_mults=dim_mults,
                device=device
            ).to(device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    model.load_state_dict(torch.load(f"path//best_model_stain_{test_id}_{name}.pth"))
    ema_model.load_state_dict(torch.load(f"path//best_model_stain_{test_id}_ema_{name}.pth"))
    model.to(device)
    ema_model.to(device)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    datasets = [DatasetDigitalStaining(img_folder, augmentation=None)]
    combined_dataset = ConcatDataset(datasets)
    loader = DataLoader(combined_dataset, batch_size=n_result, shuffle=False)
    ph1, ph2, real = next(iter(loader))
    # ph1 = torch.zeros_like(ph1)
    # ph2 = torch.zeros_like(ph2)
    return make_fake(ph1, ph2, real, model, ema_model, img_size, device, n_result,
                     mode_dif=mode_dif, noise_steps=noise_steps, noise_add=noise_add, cfg_scale=cfg_scale,
                     num_inference_steps=num_inference_steps, pred_size=pred_size, target_steps=target_steps,
                     stop_steps=stop_steps)


def make_fake(
        ph1, ph2, real, model, ema_model, img_size, device, n_result,
        mode_dif="ddim", noise_steps=1000, noise_add=True, cfg_scale=5,
        num_inference_steps=50, pred_size=256, target_steps=None,
        stop_steps=None,
):
    if mode_dif == "ddim":
        diffusion = Diffusion_ddim(noise_steps=noise_steps, img_size=img_size, device=device,
                                   noise_add=noise_add, cfg_scale=cfg_scale)
    else:
        diffusion = Diffusion_ddpm(noise_steps=noise_steps, img_size=img_size, device=device,
                                   noise_add=noise_add, cfg_scale=cfg_scale)
    x = torch.cat([ph1.to(device), ph2.to(device)], dim=1)
    if img_size is not None:
        stride = img_size // 2
        _, _, H, W = x.shape
        # H_res = H % stride
        # W_res = W % stride
        # y_range = np.arange(0, H - img_size + 1, stride)
        # x_range = np.arange(0, W - img_size + 1, stride)
        # if H_res > 0:
        #     y_range = np.append(y_range, (H - img_size))
        # if W_res > 0:
        #     x_range = np.append(x_range, (W - img_size))

        H_range = [H // 2 - pred_size // 2, H // 2 + pred_size // 2]
        W_range = [W // 2 - pred_size // 2, W // 2 + pred_size // 2]

        ph1 = ph1[:, :, H_range[0]: H_range[1], W_range[0]: W_range[1]]
        ph2 = ph2[:, :, H_range[0]: H_range[1], W_range[0]: W_range[1]]
        real = real[:, :, H_range[0]: H_range[1], W_range[0]: W_range[1]]
        y_range = np.arange(0, pred_size - img_size + 1, stride)
        x_range = np.arange(0, pred_size - img_size + 1, stride)

        # 予測結果と重みを蓄積するためのTensorをGPU上に初期化
        predictions_sum = torch.zeros_like(ph1, dtype=torch.float32).to(device)
        predictions_sum_ema = torch.zeros_like(ph1, dtype=torch.float32).to(device)
        weights_sum = torch.zeros_like(ph1, dtype=torch.float32).to(device)

        # 重み付けマップを生成 (GPU上に作成)
        blending_mask = create_gaussian_blending_mask(img_size, device)


        # 予測結果と重みを蓄積するためのTensorをGPU上に初期化
        predictions_sum = torch.zeros_like(ph1, dtype=torch.float32).to(device)
        predictions_sum_ema = torch.zeros_like(ph1, dtype=torch.float32).to(device)
        weights_sum = torch.zeros_like(ph1, dtype=torch.float32).to(device)

        # 重み付けマップを生成 (GPU上に作成)
        blending_mask = create_gaussian_blending_mask(img_size, device)

        # 推論中は勾配計算を無効化してメモリ効率を上げる
        with torch.no_grad():
            # y方向（縦）にスライド
            for y_i in tqdm(y_range):
                # x方向（横）にスライド
                for x_i in x_range:
                    # GPU上のTensorから直接パッチを切り出す
                    patch = torch.concat([
                        ph1[:, :, y_i:y_i + img_size, x_i:x_i + img_size],
                        ph2[:, :, y_i:y_i + img_size, x_i:x_i + img_size]
                    ], dim=1).to(device)

                    # モデルで予測を実行 (GPU上で計算)
                    if target_steps is not None:
                        x = real[:, :, y_i:y_i + img_size, x_i:x_i + img_size]
                        predicted_patch = diffusion.sample_target(x, model, n=n_result, labels=patch, num_inference_steps=num_inference_steps,
                                                           noise_add=noise_add, cfg_scale=cfg_scale, target_steps=target_steps)
                        predicted_patch_ema = diffusion.sample_target(x, ema_model, n=n_result, labels=patch, num_inference_steps=num_inference_steps,
                                                           noise_add=noise_add, cfg_scale=cfg_scale, target_steps=target_steps)

                    elif stop_steps is not None:
                        predicted_patch = diffusion.sample_stop(model, n=n_result, labels=patch, num_inference_steps=num_inference_steps,
                                                           noise_add=noise_add, cfg_scale=cfg_scale, stop_steps=stop_steps)
                        predicted_patch_ema = diffusion.sample_stop(ema_model, n=n_result, labels=patch, num_inference_steps=num_inference_steps,
                                                               noise_add=noise_add, cfg_scale=cfg_scale, stop_steps=stop_steps)

                    else:
                        predicted_patch = diffusion.sample(model, n=n_result, labels=patch, num_inference_steps=num_inference_steps,
                                                           noise_add=noise_add, cfg_scale=cfg_scale)
                        predicted_patch_ema = diffusion.sample(ema_model, n=n_result, labels=patch, num_inference_steps=num_inference_steps,
                                                               noise_add=noise_add, cfg_scale=cfg_scale)


                    # 予測結果と重みを対応する位置に加算 (GPU上で計算)
                    predictions_sum[:, :, y_i:y_i + img_size,
                    x_i:x_i + img_size] += predicted_patch[:, 0:1, :, :] * blending_mask
                    predictions_sum_ema[:, :, y_i:y_i + img_size,
                    x_i:x_i + img_size] += predicted_patch_ema[:, 0:1, :, :] * blending_mask
                    weights_sum[:, :, y_i:y_i + img_size, x_i:x_i + img_size] += blending_mask

        # ゼロ除算を避ける
        weights_sum[weights_sum == 0] = 1.0

        # 加重平均を計算して最終的な予測結果を得る (GPU上で計算)
        fake = (predictions_sum / weights_sum).cpu().detach().numpy()
        fake_ema = (predictions_sum_ema / weights_sum).cpu().detach().numpy()

    else:
        with torch.no_grad():
            pred = model(x)  # [1,2,img_size,img_size]
            pred_ema = ema_model(x)
        # output_pred = pred[0][0]
        fake = F.sigmoid(pred).cpu().detach().numpy()
        # output_pred = pred_ema[0][0]
        fake_ema = F.sigmoid(pred_ema).cpu().detach().numpy()
    return ph1.numpy(), ph2.numpy(), real.numpy(), fake, fake_ema
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