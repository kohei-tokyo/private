import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any


# 1. すべての設定をこのデータクラスにまとめる
@dataclass
class DigitalStainingConfig:
    """DigitalStainingの全設定を保持するクラス"""

    # --- パス設定 ---
    # モードに応じて意味が変わるパス
    dir: Optional[str] = None  # train/test: 元データ, predict: 入力画像
    new_dir: Optional[str] = None  # predict: 予測画像保存先

    # データフォルダ名
    train_folders: List[str] = field(default_factory=lambda: ["train"])
    val_folders: List[str] = field(default_factory=lambda: ["val"])
    test_folders: List[str] = field(default_factory=lambda: ["test"])

    # --- 実験管理 ---
    name: str = "Run"
    group: Optional[str] = None
    target: str = "mito" # "mito" or "ER"
    img_size: int = 256 # 128 or 256
    val_crop: bool = True
    images_to_use: str = "both"

    # --- モデルアーキテクチャ設定 ---
    encorder_name: str = "resnet34"  # "resnet34", "resnet50", or "efficientnet-b4"
    decoder_attention_type: Optional[str] = None  # 'scse' or None
    discriminator: str = "Patch4"  # Patch4, Patch3, Patch5, ResnetPatch, Resnet, or U_Net
    in_chans: int = 2

    # --- 学習パラメータ ---
    n_epoch: int = 10
    learning_rate_g: float = 0.0002
    learning_rate_d: float = 0.0002
    betas: Tuple[float, float] = (0.5, 0.999)
    batch_size: int = 16
    patches_per_epoch: int = 200
    val_epoch: int = 1
    epoch_start_ema: int = 50
    ema_beta: float = 0.9999

    # --- 損失関数の重み ---
    w_l1: float = 50.0
    w_ssim: float = 1.0
    w_dice: float = 1.0

    # --- 予測設定 ---
    test_id: str = "lpips"

    # --- 環境設定 ---
    num_workers: int = 4 # GPUのメモリが足りない場合は小さくしてください
    device: Any = field(default_factory=lambda: torch.device(
        f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'))