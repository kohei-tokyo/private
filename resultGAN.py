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

from config import DigitalStainingConfig
from make_dataset import DatasetDigitalStaining
from discriminator_modules import *
from generator_modules import *

def predict(
        img_folder,
        name="Test",
        encoder_name="resnet34",
        decoder_attention_type=None,
        test_id="lpips",
        img_size=256,
        n_result=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
):
    in_chans = 2
    class_num = 1
    G = smp.Unet(
        encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=in_chans,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=class_num,  # model output channels (number of classes in your dataset)
        decoder_attention_type=decoder_attention_type
    ).to(device)
    ema_G = copy.deepcopy(G).eval().requires_grad_(False)
    G.load_state_dict(torch.load(f"path//best_model_G_stain_{test_id}_{name}.pth"))
    ema_G.load_state_dict(torch.load(f"path//best_model_G_stain_{test_id}_ema_{name}.pth"))
    G.to(device)
    ema_G.to(device)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    datasets = [DatasetDigitalStaining(img_folder, augmentation=None)]
    combined_dataset = ConcatDataset(datasets)
    loader = DataLoader(combined_dataset, batch_size=n_result, shuffle=False)
    ph1, ph2, real = next(iter(loader))
    fake, fake_ema = make_fake(ph1, ph2, G, ema_G, img_size, device)
    return ph1.numpy(), ph2.numpy(), real.numpy(), fake, fake_ema


def make_fake(ph1, ph2, G, ema_G, img_size, device):
    x = torch.cat([ph1.to(device), ph2.to(device)], dim=1)
    if img_size is not None:
        stride = img_size // 2
        _, _, H, W = x.shape
        H_res = H % stride
        W_res = W % stride
        y_range = np.arange(0, H - img_size + 1, stride)
        x_range = np.arange(0, W - img_size + 1, stride)
        if H_res > 0:
            y_range = np.append(y_range, (H - img_size))
        if W_res > 0:
            x_range = np.append(x_range, (W - img_size))

        # 予測結果と重みを蓄積するためのTensorをGPU上に初期化
        predictions_sum = torch.zeros_like(ph1, dtype=torch.float32).to(device)
        predictions_sum_ema = torch.zeros_like(ph1, dtype=torch.float32).to(device)
        weights_sum = torch.zeros_like(ph1, dtype=torch.float32).to(device)

        # 重み付けマップを生成 (GPU上に作成)
        blending_mask = create_gaussian_blending_mask(img_size, device)

        # 推論中は勾配計算を無効化してメモリ効率を上げる
        with torch.no_grad():
            # y方向（縦）にスライド
            for y_i in y_range:
                # x方向（横）にスライド
                for x_i in x_range:
                    # GPU上のTensorから直接パッチを切り出す
                    patch = x[:, :, y_i:y_i + img_size, x_i:x_i + img_size]

                    # モデルで予測を実行 (GPU上で計算)
                    predicted_patch = F.sigmoid(G(patch))
                    predicted_patch_ema = F.sigmoid(ema_G(patch))

                    # 予測結果と重みを対応する位置に加算 (GPU上で計算)
                    predictions_sum[:, :, y_i:y_i + img_size,
                    x_i:x_i + img_size] += predicted_patch * blending_mask
                    predictions_sum_ema[:, :, y_i:y_i + img_size,
                    x_i:x_i + img_size] += predicted_patch_ema * blending_mask
                    weights_sum[:, :, y_i:y_i + img_size, x_i:x_i + img_size] += blending_mask

        # ゼロ除算を避ける
        weights_sum[weights_sum == 0] = 1.0

        # 加重平均を計算して最終的な予測結果を得る (GPU上で計算)
        fake = (predictions_sum / weights_sum).cpu().detach().numpy()
        fake_ema = (predictions_sum_ema / weights_sum).cpu().detach().numpy()

    else:
        with torch.no_grad():
            pred = G(x)  # [1,2,img_size,img_size]
            pred_ema = ema_G(x)
        # output_pred = pred[0][0]
        fake = F.sigmoid(pred).cpu().detach().numpy()
        # output_pred = pred_ema[0][0]
        fake_ema = F.sigmoid(pred_ema).cpu().detach().numpy()
    return fake, fake_ema

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

# def result(name_list=None,
#            test_id="lpips",
#            model_ema="both", # "both", "normal", or "ema"
#            n_result=1
#            ):
#     G = smp.Unet(
#         encoder_name=encorder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#         encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
#         in_channels=in_chans,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#         classes=class_num,  # model output channels (number of classes in your dataset)
#         decoder_attention_type=decoder_attention_type
#     ).to(device)
#
#     if name is None:
#         name = ["Test"]
#     loader = test_loader
#     if len(name) == 1:
#         x_len = 2
#         y_len = 1
#     else:
#         x_len = 3
#         y_len = len(name) // x_len + 1
#     n_x = 0
#     for ph1, ph2, real in loader:
#         fig, axs = plt.subplots(y_len, x_len, figsize=(x_len * 5, y_len * 5))
#         for y_i in range(y_len):
#             for x_i in range(x_len):
#                 if x_i + y_i == 0:
#                     axs[0, 0].imshow(real[0].squeeze())
#                     axs[0, 0].axis('off')
#                     axs[0, 0].set_title('target')
#                 else:
#                     G.load_state_dict(torch.load(f"path//best_model_G_stain_{test_id}_{name[n_x]}.pth"))
#                     ema_G.load_state_dict(torch.load(f"path//best_model_G_stain_{test_id}_ema_{name[n_x]}.pth"))
#                     G.to(device)
#                     ema_G.to(device)
#                     if model_ema:
#                         _, fake = make_fake(ph1, ph2)
#                     else:
#                         fake, _ = make_fake(ph1, ph2)
#                     axs[x_i, y_i].imshow(fake)
#                     axs[x_i, y_i].axis('off')
#                     axs[x_i, y_i].set_title(name[n_x])
#         plt.tight_layout()
#         plt.show()
#         n_x += 1
#         if n_x >= n_result:
#             break
