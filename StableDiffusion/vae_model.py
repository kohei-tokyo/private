import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    新コードのResBlockを移植。
    入力を出力に加算することで、層を深くしても学習を安定させる。
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 入力と出力の次元やサイズが違う場合の調整用
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out


class VAE_model(nn.Module):
    def __init__(self, in_channels=1, latent_dim=4, hidden_dims=[32, 64, 128, 256]):
        super(VAE_model, self).__init__()

        self.latent_dim = latent_dim

        # --- Encoder (ResNet構造に変更) ---
        encoder_modules = []
        curr_channels = in_channels

        for h_dim in hidden_dims:
            encoder_modules.append(
                nn.Sequential(
                    # ダウンサンプリング (Stride=2)
                    nn.Conv2d(curr_channels, h_dim, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(inplace=True),
                    # 特徴抽出 (ResBlock)
                    ResBlock(h_dim, h_dim)
                )
            )
            curr_channels = h_dim

        self.encoder = nn.Sequential(*encoder_modules)

        # 平均(mu)と分散(log_var)への変換層 (1x1 Convを使用)
        self.fc_mu = nn.Conv2d(hidden_dims[-1], latent_dim, kernel_size=1)
        self.fc_var = nn.Conv2d(hidden_dims[-1], latent_dim, kernel_size=1)

        # --- Decoder (Upsample + ResBlockに変更) ---
        self.decoder_input = nn.Conv2d(latent_dim, hidden_dims[-1], kernel_size=1)

        decoder_modules = []
        reversed_dims = hidden_dims[::-1]  # [256, 128, 64, 32]

        for i in range(len(reversed_dims) - 1):
            in_ch = reversed_dims[i]
            out_ch = reversed_dims[i + 1]

            decoder_modules.append(
                nn.Sequential(
                    # アップサンプリング (Nearest Neighbor + Conv)
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(inplace=True),
                    # 特徴の整え (ResBlock)
                    ResBlock(out_ch, out_ch)
                )
            )

        self.decoder = nn.Sequential(*decoder_modules)

        # 最後の出力層
        self.final_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(reversed_dims[-1], reversed_dims[-1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reversed_dims[-1]),
            nn.LeakyReLU(inplace=True),
            # 出力チャンネルへの変換
            nn.Conv2d(reversed_dims[-1], 1, kernel_size=3, padding=1),
            nn.Tanh()  # -1~1 に正規化
        )

    def encode(self, input):
        """画像を潜在変数へ"""
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """Reparameterization Trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """潜在変数を画像へ"""
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        # 以前のコードと同じ戻り値を維持
        return self.decode(z), input, mu, log_var

class VAE_model_old(nn.Module):
    def __init__(self, in_channels=1, latent_dim=4, hidden_dims=[32, 64, 128, 256]):
        super(VAE_model_old, self).__init__()

        self.latent_dim = latent_dim

        # --- Encoder ---
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)

        # 平均(mu)と分散(log_var)への変換層
        self.fc_mu = nn.Conv2d(hidden_dims[-1], latent_dim, kernel_size=3, padding=1)
        self.fc_var = nn.Conv2d(hidden_dims[-1], latent_dim, kernel_size=3, padding=1)

        # --- Decoder ---
        modules = []
        self.decoder_input = nn.Conv2d(latent_dim, hidden_dims[-1], kernel_size=3, padding=1)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        # 最後の出力層 (元のチャンネル数に戻す)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], 1, kernel_size=3, padding=1),  # 出力チャンネル数 (例: Mitoのみなら1)
            nn.Tanh()  # 画像が -1~1 の場合
        )

    def encode(self, input):
        """画像を潜在変数へ"""
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """Reparameterization Trick (学習時のみノイズを加える)"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """潜在変数を画像へ"""
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), input, mu, log_var