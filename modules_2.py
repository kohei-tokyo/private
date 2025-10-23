import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """
    時間tをベクトル表現に変換するためのサイン波ポジションエンベディング
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """
    ResNetブロック: (Conv -> GroupNorm -> SiLU) x 2 + Time Embedding + 残差接続
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()
        self.mlp_time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU()
        )

        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        # 時間埋め込みを処理
        time_emb = self.mlp_time(t)
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)  # (B, 2C, 1, 1)
        scale, shift = time_emb.chunk(2, dim=1)  # (B, C, 1, 1)

        # メインの処理
        h = self.block1(x)
        h = h * (scale + 1) + shift  # Affine変換で時間情報を注入
        h = self.block2(h)

        return h + self.res_conv(x)  # 残差接続


class SelfAttention(nn.Module):
    """
    Self-Attentionブロック
    """

    def __init__(self, channels, num_heads=4, head_dim=32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        inner_dim = num_heads * head_dim

        self.norm = nn.GroupNorm(32, channels)
        self.to_qkv = nn.Conv2d(channels, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x_norm = self.norm(x)
        qkv = self.to_qkv(x_norm).chunk(3, dim=1)
        q, k, v = map(
            lambda t: t.reshape(b, self.num_heads, self.head_dim, h * w), qkv
        )

        attention = torch.einsum('b h d n, b h e n -> b h d e', q, k) * self.scale
        attention = F.softmax(attention, dim=-1)

        out = torch.einsum('b h d e, b h e n -> b h d n', attention, v)

        # Reshape using the correct inner dimension
        inner_dim = self.num_heads * self.head_dim
        out = out.reshape(b, inner_dim, h, w)  # <--- FIXED LINE

        return self.to_out(out) + x


def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


# -----------------------------------------------------------------
# U-Net 本体 (Main U-Net Class)
# -----------------------------------------------------------------

class UNet_gemini(nn.Module):
    def __init__(
            self,
            c_in=6,  # ノイズ画像(3ch) + 条件画像(3ch)
            c_out=3,  # ノイズを予測
            base_dim=128,  # ベースとなるチャンネル数
            dim_mults=(1, 2, 4, 4),  # 解像度ごとのチャンネル数の倍率
            device="cuda"
    ):
        super().__init__()
        self.device = device

        # --- 時間埋め込み (Time Embedding) ---
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # --- メインのネットワーク ---
        dims = [base_dim] + [base_dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        # 初期畳み込み
        self.init_conv = nn.Conv2d(c_in, base_dim, kernel_size=3, padding=1)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # --- エンコーダ (Down path) ---
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList([
                    ResidualBlock(dim_in, dim_out, time_emb_dim=time_dim),
                    ResidualBlock(dim_out, dim_out, time_emb_dim=time_dim),
                    SelfAttention(dim_out) if ind == 2 else nn.Identity(),  # 低解像度層にAttentionを適用
                    Downsample(dim_out) if not is_last else nn.Identity(),
                ])
            )

        # --- ボトルネック (Bottleneck) ---
        mid_dim = dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = SelfAttention(mid_dim)
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        # --- デコーダ (Up path) ---
        # チャンネル数の計算ロジックをより頑健なものに修正します
        self.ups = nn.ModuleList([])
        # ボトルネックの出力から開始
        in_ch = mid_dim

        # 解像度レベルを逆順にループ (例: 512->512, 512->256, 256->128, 128->128)
        for i in reversed(range(num_resolutions)):
            is_last = i == 0

            # このレベルに対応するエンコーダからのスキップ接続のチャンネル数
            skip_ch = dims[i + 1]

            # ResBlockへの入力チャンネル数 = (下の層からの入力 + スキップ接続)
            res_in_ch = in_ch + skip_ch

            # このレベルの出力チャンネル数
            res_out_ch = dims[i]

            self.ups.append(
                nn.ModuleList([
                    ResidualBlock(res_in_ch, res_out_ch, time_emb_dim=time_dim),
                    ResidualBlock(res_out_ch, res_out_ch, time_emb_dim=time_dim),
                    SelfAttention(res_out_ch) if i > 1 else nn.Identity(),  # 低解像度(ch>=256)でAttention
                    Upsample(res_out_ch) if not is_last else nn.Identity(),
                ])
            )
            # 次のループのために、現在の出力チャンネル数を入力として設定
            in_ch = res_out_ch

        # 出力層
        # 最後のResidualBlockは不要になったため、Conv2dのみにする
        self.final_conv = nn.Conv2d(base_dim, c_out, 1)

    def forward(self, x_noisy, time, cond_image=None):  # cond_imageにデフォルト値Noneを設定
        # 1. 入力と条件画像を結合
        if cond_image is not None:
            # --- 条件付きの場合 (Conditional) ---
            x = torch.cat([x_noisy, cond_image], dim=1)
        else:
            # --- 条件なしの場合 (Unconditional) ---
            # 条件画像のチャンネル数分のゼロパディングを作成する
            # 例: init_convの入力が6ch, x_noisyが3chなら、3ch分のゼロを作る
            cond_channels = self.init_conv.in_channels - x_noisy.shape[1]
            zeros = torch.zeros(x_noisy.shape[0], cond_channels, x_noisy.shape[2], x_noisy.shape[3],
                                device=x_noisy.device)
            x = torch.cat([x_noisy, zeros], dim=1)

        # 2. 時間埋め込みを計算
        t = self.time_mlp(time)

        # 3. 初期畳み込み
        x = self.init_conv(x)

        # スキップ接続用の特徴マップを保持
        h = []

        # 4. エンコーダ
        for resnet1, resnet2, attn, downsample in self.downs:
            x = resnet1(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # 5. ボトルネック
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # 6. デコーダ
        for resnet1, resnet2, attn, upsample in self.ups:
            # スキップ接続と結合
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet1(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        # 7. 出力
        return self.final_conv(x)  # 最後の畳み込み層を呼び出す


# class Unet(nn.Module):
#     def __init__(
#             self, c_in=2+1, c_out=1, time_dim=256, num_classes=None, device="cuda",
#             dim_mults=(1, 2, 4, 4)  # 解像度ごとのチャンネル数の倍率
#     ):
#         super().__init__()
#         self.device = device
#         self.time_dim = time_dim
#         base_dim = time_dim // 4
#
#         # --- 時間埋め込み (Time Embedding) ---
#         self.time_mlp = nn.Sequential(
#             SinusoidalPositionEmbeddings(base_dim),
#             nn.Linear(base_dim, time_dim),
#             nn.GELU(),
#             nn.Linear(time_dim, time_dim),
#         )
#
#         # --- メインのネットワーク ---
#         dims = [base_dim] + [base_dim * m for m in dim_mults]
#         in_out = list(zip(dims[:-1], dims[1:]))
#
#         # 初期畳み込み
#         self.init_conv = nn.Conv2d(c_in, base_dim, kernel_size=3, padding=1)
#
#         self.downs = nn.ModuleList([])
#         self.ups = nn.ModuleList([])
#         num_resolutions = len(in_out)
#
#         # --- エンコーダ (Down path) ---
#         for ind, (dim_in, dim_out) in enumerate(in_out):
#             is_last = ind >= (num_resolutions - 1)
#             self.downs.append(
#                 nn.ModuleList([
#                     ResidualBlock(dim_in, dim_out, time_emb_dim=time_dim),
#                     ResidualBlock(dim_out, dim_out, time_emb_dim=time_dim),
#                     SelfAttention(dim_out) if ind == 2 else nn.Identity(),  # 低解像度層にAttentionを適用
#                     Downsample(dim_out) if not is_last else nn.Identity(),
#                 ])
#             )
#
#         # --- ボトルネック (Bottleneck) ---
#         mid_dim = dims[-1]
#         self.mid_block1 = ResidualBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
#         self.mid_attn = SelfAttention(mid_dim)
#         self.mid_block2 = ResidualBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
#
#         # --- デコーダ (Up path) ---
#         for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
#             is_last = ind >= (num_resolutions - 2)
#             self.ups.append(
#                 nn.ModuleList([
#                     # スキップ接続からのチャンネルを考慮 (dim_out*2)
#                     ResidualBlock(dim_out * 2, dim_in, time_emb_dim=time_dim),
#                     ResidualBlock(dim_in, dim_in, time_emb_dim=time_dim),
#                     SelfAttention(dim_in) if ind == 1 else nn.Identity(),
#                     Upsample(dim_in) if not is_last else nn.Identity(),
#                 ])
#             )
#
#         # 出力層
#         self.final_conv = nn.Sequential(
#             ResidualBlock(base_dim, base_dim, time_emb_dim=time_dim),
#             nn.Conv2d(base_dim, c_out, 1)
#         )
#
#     def forward(self, x_noisy, time, cond_image):
#         # 1. 入力と条件画像を結合
#         x = torch.cat([x_noisy, cond_image], dim=1)
#
#         # 2. 時間埋め込みを計算
#         t = self.time_mlp(time)
#
#         # 3. 初期畳み込み
#         x = self.init_conv(x)
#
#         # スキップ接続用の特徴マップを保持
#         h = []
#
#         # 4. エンコーダ
#         for resnet1, resnet2, attn, downsample in self.downs:
#             x = resnet1(x, t)
#             x = resnet2(x, t)
#             x = attn(x)
#             h.append(x)
#             x = downsample(x)
#
#         # 5. ボトルネック
#         x = self.mid_block1(x, t)
#         x = self.mid_attn(x)
#         x = self.mid_block2(x, t)
#
#         # 6. デコーダ
#         for resnet1, resnet2, attn, upsample in self.ups:
#             # スキップ接続と結合
#             x = torch.cat((x, h.pop()), dim=1)
#             x = resnet1(x, t)
#             x = resnet2(x, t)
#             x = attn(x)
#             x = upsample(x)
#
#         # 7. 出力
#         return self.final_conv(x)
#
#     def pos_encoding(self, t, channels):
#         inv_freq = 1.0 / (
#             10000
#             ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
#         )
#         pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
#         pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
#         pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
#         return pos_enc
# -----------------------------------------------------------------
# 実行テスト (Execution Test)
# -----------------------------------------------------------------

if __name__ == '__main__':
    # --- モデルのインスタンス化 ---
    model = Unet(
        in_channels=6,  # ノイズ画像(3) + 条件画像(3)
        out_channels=3,
        base_dim=128,
        dim_mults=(1, 2, 4, 4)
    )

    # --- ダミーデータの作成 ---
    batch_size = 2
    image_size = 128

    # ノイズが加えられた画像 (ターゲット)
    dummy_noisy_image = torch.randn(batch_size, 3, image_size, image_size)

    # 条件となる画像
    dummy_cond_image = torch.randn(batch_size, 3, image_size, image_size)

    # タイムステップ (0から999などの整数)
    dummy_time = torch.randint(0, 1000, (batch_size,)).long()

    # --- モデルのフォワードパスを実行 ---
    predicted_noise = model(dummy_noisy_image, dummy_time, dummy_cond_image)

    # --- 出力サイズの確認 ---
    print(f"Input shape: {dummy_noisy_image.shape}")
    print(f"Predicted noise shape: {predicted_noise.shape}")

    # --- 入力と出力の形状が同じであることを確認 ---
    assert predicted_noise.shape == dummy_noisy_image.shape
    print("\n✅ Test passed! The output shape is correct.")