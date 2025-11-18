import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import segmentation_models_pytorch as smp


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


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

    def __init__(self, in_channels, out_channels, time_emb_dim, GN=True, groups=8):
        super().__init__()
        self.mlp_time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels) if GN else nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels) if GN else nn.BatchNorm2d(out_channels),
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
            c_in=2,  # ノイズ画像(3ch) + 条件画像(3ch)
            c_out=1,  # ノイズを予測
            base_dim=64,  # ベースとなるチャンネル数
            dim_mults=(1, 2, 4, 4),  # 解像度ごとのチャンネル数の倍率
            num_embeddings=2,
            self_attension=True,
            device="cuda"
    ):
        super().__init__()
        self.device = device

        # --- 時間埋め込み (Time Embedding) ---
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            # SinusoidalPositionEmbeddings(base_dim),
            nn.Embedding(num_embeddings=num_embeddings, embedding_dim=base_dim),
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
                    SelfAttention(dim_out) if (ind == 2 and self_attension) else nn.Identity(),
                    Downsample(dim_out) if not is_last else nn.Identity(),
                ])
            )

        # --- ボトルネック (Bottleneck) ---
        mid_dim = dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = SelfAttention(mid_dim) if self_attension else nn.Identity()
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
                    SelfAttention(res_out_ch) if (i > 1 and self_attension) else nn.Identity(),  # 低解像度(ch>=256)でAttention
                    Upsample(res_out_ch) if not is_last else nn.Identity(),
                ])
            )
            # 次のループのために、現在の出力チャンネル数を入力として設定
            in_ch = res_out_ch

        # 出力層
        # 最後のResidualBlockは不要になったため、Conv2dのみにする
        self.final_conv = nn.Conv2d(base_dim, c_out, 1)

    def forward(self, x, t):
        # 1. 入力と条件画像を結合
        # 2. 時間埋め込みを計算
        t = self.time_mlp(t)

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

        #for feature in h:
            #print(feature.shape)

        # 6. デコーダ
        for resnet1, resnet2, attn, upsample in self.ups:
            # スキップ接続と結合
            # print(x.shape)
            # print(h.pop().shape)
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet1(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        # 7. 出力
        return self.final_conv(x)  # 最後の畳み込み層を呼び出す



class UNet_gemini_pretrained_ori(nn.Module):
    def __init__(
            self,
            c_in=2,  # ノイズ画像(3ch) + 条件画像(3ch)
            c_out=1,  # ノイズを予測
            num_embeddings=2,
            encoder_name="resnet34",
            self_attension=True,
            device="cuda"
    ):
        super().__init__()
        self.device = device

        # --- エンコーダ (Down path) ---
        self.downs = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=c_in,
            classes=c_out  # (このc_outはダミー)
        ).encoder

        #out_channels = self.downs.out_channels[2:]
        #base_dim = out_channels[0]
        #dim_mults = [out_channel // base_dim for out_channel in out_channels]
        #print("dim_mults:", dim_mults)
        #dims = [base_dim] + [base_dim * m for m in dim_mults]
        #print("dims:", dims)
        #in_out = list(zip(dims[:-1], dims[1:]))
        dims = self.downs.out_channels
        base_dim = dims[1]

        # --- 時間埋め込み (Time Embedding) ---
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            # SinusoidalPositionEmbeddings(base_dim),
            nn.Embedding(num_embeddings=num_embeddings, embedding_dim=base_dim),
            nn.Linear(base_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.ups = nn.ModuleList([])
        num_resolutions = len(dims) - 1

        # --- ボトルネック (Bottleneck) ---
        mid_dim = dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = SelfAttention(mid_dim) if self_attension else nn.Identity()
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        # --- デコーダ (Up path) ---
        # チャンネル数の計算ロジックをより頑健なものに修正します
        self.ups = nn.ModuleList([])
        # ボトルネックの出力から開始
        in_ch = mid_dim
        if encoder_name == "resnet34":
            decoder_channels = [32, 64, 64, 128, 256]
        elif encoder_name == "efficientnet-b4":
            decoder_channels = [16, 32, 64, 128, 256]
        else:
            decoder_channels = dims

        # 解像度レベルを逆順にループ (例: 512->512, 512->256, 256->128, 128->128)
        for i in reversed(range(num_resolutions)):
            is_last = i == 0

            # このレベルに対応するエンコーダからのスキップ接続のチャンネル数
            skip_ch = dims[i]

            # ResBlockへの入力チャンネル数 = (下の層からの入力 + スキップ接続)
            res_in_ch = in_ch + skip_ch

            # このレベルの出力チャンネル数
            # res_out_ch = dims[i - 1] if not is_last else dims[i - 2]
            res_out_ch = decoder_channels[i]

            self.ups.append(
                nn.ModuleList([
                    Upsample(in_ch),
                    ResidualBlock(res_in_ch, res_out_ch, time_emb_dim=time_dim, GN=True if not is_last else False),
                    ResidualBlock(res_out_ch, res_out_ch, time_emb_dim=time_dim, GN=True if not is_last else False),
                    SelfAttention(res_out_ch) if (i > 1 and self_attension) else nn.Identity(),  # 低解像度(ch>=256)でAttention
                    # Upsample(res_out_ch) if not is_last else nn.Identity(),
                ])
            )
            # 次のループのために、現在の出力チャンネル数を入力として設定
            in_ch = res_out_ch

        # 出力層
        # 最後のResidualBlockは不要になったため、Conv2dのみにする
        self.final_conv = nn.Conv2d(in_ch, c_out, 1)

    def forward(self, x, t):
        # 1. 入力と条件画像を結合
        # 2. 時間埋め込みを計算
        t = self.time_mlp(t)

        # スキップ接続用の特徴マップを保持
        features = self.downs(x)
        x = features[-1]
        h = features[0:-1]

        # 5. ボトルネック
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # 6. デコーダ
        for upsample, resnet1, resnet2, attn in self.ups:
            x = upsample(x)
            # スキップ接続と結合
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet1(x, t)
            x = resnet2(x, t)
            x = attn(x)

        # 7. 出力
        return self.final_conv(x)  # 最後の畳み込み層を呼び出す


class UNet_gemini_pretrained(nn.Module):
    def __init__(
            self,
            c_in=2,  # 入力チャンネル数 (例: ph1 + ph2)
            c_out=1,  # 出力チャンネル数 (例: mito)
            num_embeddings=2,  # 条件のクラス数
            self_attension=True,
            encoder_name="resnet34",
            device="cuda"
    ):
        """
        smpの事前学習済みエンコーダと、条件付け（FiLM）が可能な
        カスタムデコーダを組み合わせたハイブリッドU-Net
        """
        super().__init__()
        self.device = device

        # --- 1. 事前学習済みエンコーダのロード ---
        self.encoder = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=c_in,
            classes=c_out  # (このc_outはダミー)
        ).encoder

        # エンコーダの各ステージの出力チャンネル数を取得
        encoder_channels = self.encoder.out_channels
        # 例 (resnet34): [3, 64, 64, 128, 256, 512]

        # --- 2. 時間 (条件) 埋め込み ---
        # 条件ベクトル (t) の次元数を定義
        time_dim = 256  # FiLM (scale/shift) 生成用の次元
        embedding_dim = 128  # nn.Embedding の中間次元
        self.time_mlp = nn.Sequential(
            nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim),
            nn.Linear(embedding_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # --- 3. ボトルネック (Bottleneck) ---
        # エンコーダの最後の出力 (例: 512ch) を受け取る
        mid_dim = encoder_channels[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = SelfAttention(mid_dim) if self_attension else nn.Identity()
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        # --- 4. デコーダ (Up path) ---
        self.ups = nn.ModuleList([])

        # デコーダの各ブロックの出力チャンネル数を定義
        # (resnet34の標準的なU-Netデコーダ構成に合わせる)
        decoder_out_channels = [256, 128, 64, 64]

        # スキップ接続で受け取るチャンネル (エンコーダから逆順)
        # encoder_channels[1:-1] は [64, 64, 128, 256]
        skip_channels = list(reversed(encoder_channels[1:-1]))  # [256, 128, 64, 64]

        in_ch = mid_dim  # ボトルネック (512ch) からスタート

        for i in range(len(decoder_out_channels)):
            is_last = i == (len(decoder_out_channels) - 1)

            skip_ch = skip_channels[i]
            res_in_ch = in_ch + skip_ch  # (Up + Skip)
            res_out_ch = decoder_out_channels[i]

            self.ups.append(
                nn.ModuleList([
                    ResidualBlock(res_in_ch, res_out_ch, time_emb_dim=time_dim),
                    ResidualBlock(res_out_ch, res_out_ch, time_emb_dim=time_dim),
                    SelfAttention(res_out_ch) if (i < 2 and self_attension) else nn.Identity(),

                    # ★ 修正点 (Bug 4) ★
                    # Upsample層は、このブロックの「入力」チャンネル数 (in_ch) で初期化する
                    # (res_out_ch ではない)
                    Upsample(in_ch)
                    #  if not is_last else nn.Identity(),
                ])
            )
            in_ch = res_out_ch  # 次のブロックのために in_ch を更新

        # --- 5. 出力層 ---
        # 最後のデコーダブロックの出力 (64ch) を受け取る
        self.final_conv = nn.Conv2d(in_ch, c_out, 1)

    def forward(self, x, t):
        # 1. 時間 (条件) 埋め込みを計算
        t = self.time_mlp(t)

        # 2. エンコーダ (smp.encoder)
        # features は [input, stem, layer1, layer2, layer3, layer4] の順
        # 解像度は [H, H/2, H/4, H/8, H/16, H/32]
        features = self.encoder(x)

        # 3. エンコーダ出力を整理
        x = features[-1]  # ボトルネックへの入力 (layer4, 例: H/32)

        # ★ 修正点 (Bug 2) ★
        # reversed() を削除。h = [stem, layer1, layer2, layer3]
        # h.pop() で layer3 (H/16) から順に取り出せる
        h = features[1:-1]

        # 4. ボトルネック
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # 5. デコーダ
        for resnet1, resnet2, attn, upsample in self.ups:
            # ★ 修正点 (Bug 3) ★
            # 結合(cat) の *前* に x をアップサンプルする
            # 例: x (H/32) -> upsample(x) (H/16)
            x = upsample(x)

            # スキップ接続と結合
            # h.pop() で layer3 (H/16) が取り出される
            # (H/16) と (H/16) のサイズが一致
            x = torch.cat((x, h.pop()), dim=1)

            # ResBlock (条件注入)
            x = resnet1(x, t)
            x = resnet2(x, t)
            x = attn(x)

        # 6. 出力
        return self.final_conv(x)