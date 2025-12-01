import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional


# 1. 全ての設定をこの設定クラスにきれいにまとめます
@dataclass
class DDPMConfig:
    """DDPMモデルと学習パイプラインの全設定を保持するクラス"""

    # --- パスとデータに関する設定 ---
    dir: str
    train_folders: List[str] = field(default_factory=lambda: ["train"])
    val_folders: List[str] = field(default_factory=lambda: ["val"])
    test_folders: List[str] = field(default_factory=lambda: ["test"])
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
    w_ssim: float = 0.01
    w_target_ssim: float = 0.01
    w_target_lpips: float = 0.01
    t_target: int = 1000
    lr_epoch: int = 0

    # --- UNetモデルのアーキテクチャ設定 ---
    model_unet: str = "original"
    in_chans: int = 2
    dim_mults: Tuple[int, ...] = (1, 2, 4, 4)
    real_pred: bool = False

    # --- 拡散過程に関する設定 ---
    noise_steps: int = 1000
    noise_add: bool = True

    # --- 推論に関する設定 ---
    img_size: int = 128
    pred_size: int = 512
    num_inference_steps: int = 30
    mode_dif: str = "ddim"
    cfg_scale: int = 3
    no_label: float = 0.1
    val_crop: float = 2

    # --- 環境設定 ---
    num_workers: int = 8
    device: Any = field(default_factory=lambda: torch.device(
        f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'))

    # --- VAE ---
    vae_pth: str = "vae.pth"
    latent_dim: int = 4
    hidden_dims: List[int] = field(default_factory=lambda: [32, 64, 128, 256])


@dataclass
class VAEConfig:
    """VAEモデルと学習パイプラインの全設定を保持するクラス"""

    # --- パスとデータに関する設定 ---
    dir: str
    train_folders: List[str] = field(default_factory=lambda: ["train"])
    val_folders: List[str] = field(default_factory=lambda: ["val"])
    test_folders: List[str] = field(default_factory=lambda: ["test"])
    images_to_use: str = "both"
    specialize: str = None

    # --- 実験管理 (W&Bなど) ---
    name: str = "Test_vae"
    group: Optional[str] = None

    # --- 学習ループに関する設定 ---
    n_epoch: int = 10
    learning_rate: float = 1e-3
    batch_size: int = 16
    patches_per_epoch: int = 200
    val_epoch: int = 1
    w_kld: float = 0.00001
    img_size: int = 128

    # --- モデルのアーキテクチャ設定 ---
    latent_dim: int = 4
    hidden_dims: List[int] = field(default_factory=lambda: [32, 64, 128, 256])

    # --- 環境設定 ---
    num_workers: int = 8
    device: Any = field(default_factory=lambda: torch.device(
        f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'))