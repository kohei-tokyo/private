import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional


# 1. 全ての設定をこの設定クラスにきれいにまとめます
@dataclass
class Config:

    # --- パスとデータに関する設定 ---
    dir: List[str] = field(default_factory=lambda: [r"C:\data\mito_crop_128",r"C:\data\ER_crop_128"])
    train_folders: List[List[str]] = field(default_factory=lambda: [["2", "3", "4"], ["1"]])
    val_folders: List[List[str]] = field(default_factory=lambda: [["1"], ["2"]])
    test_folders: List[List[str]] = field(default_factory=lambda: [["5"], ["3"]])
    target: List[str] = field(default_factory=lambda: ["mito", "ER"])
    images_to_use: str = "both"

    # --- 実験管理 (W&Bなど) ---
    name: str = "Run"
    group: Optional[str] = None

    # --- 学習ループに関する設定 ---
    n_epoch: int = 10
    learning_rate: float = 3e-4
    batch_size: int = 16
    patches_per_epoch: int = 200
    patches_per_epoch_val: int = 5
    val_epoch: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate_scheduler: bool = True
    # w_ssim: float = 0.01

    # # --- UNetモデルのアーキテクチャ設定 ---
    # model_unet: str = "original"
    # in_chans: int = 2
    # dim_mults: Tuple[int, ...] = (1, 2, 4, 4)
    # real_pred: bool = False

    # --- 拡散過程に関する設定 ---
    noise_steps: int = 1000
    # noise_add: bool = True

    # --- 推論に関する設定 ---
    img_size: int = 128
    # pred_size: int = 512
    num_inference_steps: int = 30
    # mode_dif: str = "ddim"
    # cfg_scale: int = 3
    # no_label: float = 0.1
    val_crop: int = 2

    # --- 環境設定 ---
    num_workers: int = 8
    device: Any = field(default_factory=lambda: torch.device(
        f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'))