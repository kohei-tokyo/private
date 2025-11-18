import torch.nn as nn
import timm
import torch
from torch.nn.utils import spectral_norm


class Patch(nn.Module):
    def __init__(self, in_channels=3, depth=3, base_dim=64):
        super(Patch, self).__init__()

        def block(in_f, out_f, normalize=True):
            """Conv → (BN) → LeakyReLU"""
            layers = [nn.Conv2d(in_f, out_f, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.GroupNorm(num_groups=8, num_channels=out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        model_layers = []
        model_layers.extend(block(in_channels, base_dim, normalize=False))
        for i in range(depth - 1):
            model_layers.extend(block(base_dim * (2 ** i), base_dim * (2 ** (i + 1))))
        model_layers.append(nn.Conv2d(base_dim * (2 ** (depth - 1)), 1, 4, padding=1))
        self.model = nn.Sequential(*model_layers)

    def forward(self, img, c):
        return self.model(img)  # "パッチごと" の真偽スコア

class ResnetPatch(nn.Module):
    def __init__(self, in_channels=3):
        super(ResnetPatch, self).__init__()
        resnet = timm.create_model("resnet18", in_chans=in_channels, pretrained=False, num_classes=1)
        modules = list(resnet.children())[:-3]  # layer3の後まで
        self.model = nn.Sequential(
            *modules,
            nn.Conv2d(256, 1, kernel_size=1)
        )

    def forward(self, img, c):
        return self.model(img)


# discriminator_modules.py の Patch クラスの修正案
import torch.nn.functional as F


class Patch_projection(nn.Module):
    def __init__(self, in_channels=3, depth=3, base_dim=64, num_classes=2, embedding_dim=128):
        super(Patch_projection, self).__init__()

        # --- 画像パス (元々のモデル) ---
        def block(in_f, out_f, normalize=True):
            layers = [nn.Conv2d(in_f, out_f, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.GroupNorm(num_groups=8, num_channels=out_f))
            # layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        model_layers = []
        model_layers.extend(block(in_channels, base_dim, normalize=False))
        for i in range(depth - 1):
            model_layers.extend(block(base_dim * (2 ** i), base_dim * (2 ** (i + 1))))

        # 最後のConvを画像ベクトル抽出用に変更 (出力は 1 ではなく embedding_dim に)
        self.image_path = nn.Sequential(*model_layers)
        final_ch = base_dim * (2 ** (depth - 1))
        self.image_projection = nn.Conv2d(final_ch, embedding_dim, 4, padding=1)

        # --- 条件パス (c をベクトル化) ---
        self.condition_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=embedding_dim)

        # --- 本物/偽物判定用の追加レイヤー ---
        self.real_fake_output = nn.Conv2d(final_ch, 1, 4, padding=1)

    def forward(self, img, c):
        # 1. 画像から特徴を抽出
        img_features = self.image_path(img)  # (B, final_ch, H, W)

        # 2. 「本物/偽物」スコア
        score_real_fake = self.real_fake_output(img_features)  # (B, 1, H, W)

        # 3. 「マッチング」スコア (射影)
        # 画像ベクトルを生成
        img_vector = self.image_projection(img_features)  # (B, embedding_dim, H, W)

        # 条件ベクトルを生成し、画像サイズに合わせる
        cond_vector = self.condition_embedding(c)  # (B, embedding_dim)
        cond_vector = cond_vector.unsqueeze(-1).unsqueeze(-1)  # (B, embedding_dim, 1, 1)
        cond_vector_expanded = cond_vector.expand_as(img_vector)  # (B, embedding_dim, H, W)

        # 内積を計算 (パッチごと)
        score_matching = (img_vector * cond_vector_expanded).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # 4. 最終スコア
        return score_real_fake + score_matching



class Patch_projection_spectral(nn.Module):
    def __init__(self, in_channels=3, depth=3, base_dim=64, num_classes=2, embedding_dim=128):
        super(Patch_projection_spectral, self).__init__()

        # --- 画像パス (元々のモデル) ---
        def block(in_f, out_f, normalize=True):
            layers = [spectral_norm(nn.Conv2d(in_f, out_f, 4, stride=2, padding=1))]
            if normalize:
                layers.append(nn.GroupNorm(num_groups=8, num_channels=out_f))
            # layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        model_layers = []
        model_layers.extend(block(in_channels, base_dim, normalize=False))
        for i in range(depth - 1):
            model_layers.extend(block(base_dim * (2 ** i), base_dim * (2 ** (i + 1))))

        # 最後のConvを画像ベクトル抽出用に変更 (出力は 1 ではなく embedding_dim に)
        self.image_path = nn.Sequential(*model_layers)
        final_ch = base_dim * (2 ** (depth - 1))
        self.image_projection = spectral_norm(nn.Conv2d(final_ch, embedding_dim, 4, padding=1))

        # --- 条件パス (c をベクトル化) ---
        self.condition_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=embedding_dim)

        # --- 本物/偽物判定用の追加レイヤー ---
        self.real_fake_output = spectral_norm(nn.Conv2d(final_ch, 1, 4, padding=1))

    def forward(self, img, c):
        # 1. 画像から特徴を抽出
        img_features = self.image_path(img)  # (B, final_ch, H, W)

        # 2. 「本物/偽物」スコア
        score_real_fake = self.real_fake_output(img_features)  # (B, 1, H, W)

        # 3. 「マッチング」スコア (射影)
        # 画像ベクトルを生成
        img_vector = self.image_projection(img_features)  # (B, embedding_dim, H, W)

        # 条件ベクトルを生成し、画像サイズに合わせる
        cond_vector = self.condition_embedding(c)  # (B, embedding_dim)
        cond_vector = cond_vector.unsqueeze(-1).unsqueeze(-1)  # (B, embedding_dim, 1, 1)
        cond_vector_expanded = cond_vector.expand_as(img_vector)  # (B, embedding_dim, H, W)

        # 内積を計算 (パッチごと)
        score_matching = (img_vector * cond_vector_expanded).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # 4. 最終スコア
        return score_real_fake + score_matching