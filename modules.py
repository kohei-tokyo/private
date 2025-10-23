
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # print(f"x shape in sa {x.shape}")
        B, C, H, W = x.shape
        x = x.view(-1, self.channels, H*W).swapaxes(1, 2)
        # print(f"x shape in sa 2 {x.shape}")
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, H, W)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        # print(x.shape)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        # print(emb.shape)
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class UNet_conditional(nn.Module):
    def __init__(self, c_in=2+1, c_out=1, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        # Encoder (Downsampling Path)
        self.enc1 = DoubleConv(c_in, 64)      # Output: 256x256, 64 channels
        self.down1 = Down(64, 128)           # Output: 128x128, 128 channels
        self.enc2 = DoubleConv(128, 256)     # Output: 128x128, 256 channels
        self.down2 = Down(256, 512)          # Output: 64x64, 512 channels
        self.enc3 = DoubleConv(512, 512)     # Output: 64x64, 512 channels
        self.down3 = Down(512, 1024)         # Output: 32x32, 1024 channels
        self.sa1 = SelfAttention(1024)

        self.down4 = Down(1024, 2048)  # 16*16 2048 channels
        self.sa2 = SelfAttention(2048)

        self.bot1 = DoubleConv(2048, 1024)

        self.up1 = Up(2048, 512)  # 32*32
        self.sa3 = SelfAttention(512)

        self.up2 = Up(1024, 512)  # 64
        self.dec3 = DoubleConv(512, 256)
        self.up3 = Up(512, 128)  # 128

        self.dec4 = DoubleConv(128, 64)
        self.up4 = Up(128, 32)
        self.outc = nn.Conv2d(32, c_out, kernel_size=1)

        # if num_classes is not None:
        #     self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, c):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # if y is not None:
            # t += self.label_emb(y)
            # t += y
        # if self.conditional_method == "concat":
        #     t = torch.cat([t, y], dim=1)
        # else:
        #     t += y

        if c is not None:
            x0 = torch.cat([x, c], dim=1)
        else:
            y_channels = self.enc1.double_conv[0].in_channels - x.shape[1]
            zero_padding = torch.zeros(x.shape[0], y_channels, x.shape[2], x.shape[3], device=self.device)
            x0 = torch.cat([x, zero_padding], dim=1)

        x1 = self.enc1(x0)
        x2 = self.down1(x1, t)
        x2 = self.enc2(x2)
        x3 = self.down2(x2, t)
        x3 = self.enc3(x3)
        x4 = self.down3(x3, t)

        x4 = self.sa1(x4)
        x5 = self.down4(x4, t)
        x5 = self.sa2(x5)

        x = self.bot1(x5)

        x_up4 = self.up1(x, x4, t)
        x = self.sa3(x_up4)
        x = self.up2(x, x3, t)
        x = self.dec3(x)
        x = self.up3(x, x2, t)
        x = self.dec4(x)
        x = self.up4(x, x1, t)
        x = self.outc(x)
        return x

class UNet_conditional_ori(nn.Module):
    def __init__(self, c_in=2+1, c_out=1, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, c):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # if y is not None:
            # t += self.label_emb(y)
            # t += y
        # if self.conditional_method == "concat":
        #     t = torch.cat([t, y], dim=1)
        # else:
        #     t += y

        if c is not None:
            x0 = torch.cat([x, c], dim=1)
        else:
            y_channels = self.inc.double_conv[0].in_channels - x.shape[1]
            zero_padding = torch.zeros(x.shape[0], y_channels, x.shape[2], x.shape[3], device=self.device)
            x0 = torch.cat([x, zero_padding], dim=1)

        x1 = self.inc(x0)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output

class UNet_conditional_deep(nn.Module):
    def __init__(self, c_in=2+1, c_out=1, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 512)
        self.sa3 = SelfAttention(512)
        self.down4 = Down(512, 512)
        self.sa4 = SelfAttention(512)

        self.bot1 = DoubleConv(512, 1024)
        self.bot2 = DoubleConv(1024, 1024)
        self.bot3 = DoubleConv(1024, 512)

        self.up1 = Up(1024, 256)
        self.sa5 = SelfAttention(256)
        self.up2 = Up(512, 128)
        self.sa6 = SelfAttention(128)
        self.up3 = Up(256, 64)
        self.sa7 = SelfAttention(64)
        self.up4 = Up(128, 64)
        self.sa8 = SelfAttention(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, c):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # if y is not None:
            # t += self.label_emb(y)
            # t += y
        # if self.conditional_method == "concat":
        #     t = torch.cat([t, y], dim=1)
        # else:
        #     t += y

        if c is not None:
            x0 = torch.cat([x, c], dim=1)
        else:
            y_channels = self.inc.double_conv[0].in_channels - x.shape[1]
            zero_padding = torch.zeros(x.shape[0], y_channels, x.shape[2], x.shape[3], device=self.device)
            x0 = torch.cat([x, zero_padding], dim=1)

        x1 = self.inc(x0)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        x5 = self.down4(x4, t)
        x5 = self.sa4(x5)

        x5 = self.bot1(x5)
        x5 = self.bot2(x5)
        x5 = self.bot3(x5)

        x = self.up1(x5, x4, t)
        x = self.sa5(x)
        x = self.up2(x, x3, t)
        x = self.sa6(x)
        x = self.up3(x, x2, t)
        x = self.sa7(x)
        x = self.up4(x, x1, t)
        x = self.sa8(x)
        output = self.outc(x)
        return output

class UNet_conditional_deep(nn.Module):
    def __init__(self, c_in=2+1, c_out=1, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 512)
        self.sa3 = SelfAttention(512)
        self.down4 = Down(512, 512)
        self.sa4 = SelfAttention(512)

        self.bot1 = DoubleConv(512, 1024)
        self.bot2 = DoubleConv(1024, 1024)
        self.bot3 = DoubleConv(1024, 512)

        self.up1 = Up(1024, 256)
        self.sa5 = SelfAttention(256)
        self.up2 = Up(512, 128)
        self.sa6 = SelfAttention(128)
        self.up3 = Up(256, 64)
        self.sa7 = SelfAttention(64)
        self.up4 = Up(128, 64)
        self.sa8 = SelfAttention(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, c):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # if y is not None:
            # t += self.label_emb(y)
            # t += y
        # if self.conditional_method == "concat":
        #     t = torch.cat([t, y], dim=1)
        # else:
        #     t += y

        if c is not None:
            x0 = torch.cat([x, c], dim=1)
        else:
            y_channels = self.inc.double_conv[0].in_channels - x.shape[1]
            zero_padding = torch.zeros(x.shape[0], y_channels, x.shape[2], x.shape[3], device=self.device)
            x0 = torch.cat([x, zero_padding], dim=1)

        x1 = self.inc(x0)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        x5 = self.down4(x4, t)
        x5 = self.sa4(x5)

        x5 = self.bot1(x5)
        x5 = self.bot2(x5)
        x5 = self.bot3(x5)

        x = self.up1(x5, x4, t)
        x = self.sa5(x)
        x = self.up2(x, x3, t)
        x = self.sa6(x)
        x = self.up3(x, x2, t)
        x = self.sa7(x)
        x = self.up4(x, x1, t)
        x = self.sa8(x)
        output = self.outc(x)
        return output
class UNet_conditional_sa_5(nn.Module):
    def __init__(self, c_in=2+1, c_out=1, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.dc1 = DoubleConv(128, 128)
        self.down2 = Down(128, 256)
        self.sa1 = SelfAttention(256)
        self.down3 = Down(256, 512)
        self.sa2 = SelfAttention(256)
        self.down4 = Down(512, 1024)
        self.sa3 = SelfAttention(512)
        self.down5 = Down(1024, 1024)
        self.sa4 = SelfAttention(1024)

        self.bot1 = DoubleConv(1024, 2048)
        self.bot2 = DoubleConv(2048, 2048)
        self.bot3 = DoubleConv(2048, 1024)

        self.up1 = Up(2048, 512)
        self.sa5 = SelfAttention(512)
        self.up2 = Up(1024, 256)
        self.sa6 = SelfAttention(256)
        self.up3 = Up(512, 128)
        self.sa7 = SelfAttention(128)
        self.up4 = Up(256, 64)
        self.sa8 = SelfAttention(64)
        self.up5 = Up(128, 64)
        self.sa9 = SelfAttention(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, c):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # if y is not None:
            # t += self.label_emb(y)
            # t += y
        # if self.conditional_method == "concat":
        #     t = torch.cat([t, y], dim=1)
        # else:
        #     t += y

        if c is not None:
            x0 = torch.cat([x, c], dim=1)
        else:
            y_channels = self.inc.double_conv[0].in_channels - x.shape[1]
            zero_padding = torch.zeros(x.shape[0], y_channels, x.shape[2], x.shape[3], device=self.device)
            x0 = torch.cat([x, zero_padding], dim=1)

        x1 = self.inc(x0)
        x2 = self.down1(x1, t)
        x2 = self.dc1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa1(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa2(x4)
        x5 = self.down4(x4, t)
        x5 = self.sa3(x5)
        x6 = self.down5(x5, t)
        x6 = self.sa4(x6)

        x6 = self.bot1(x6)
        x6 = self.bot2(x6)
        x6 = self.bot3(x6)

        x = self.up1(x6, x5, t)
        x = self.sa5(x)
        x = self.up2(x, x4, t)
        x = self.sa6(x)
        x = self.up3(x, x3, t)
        x = self.sa7(x)
        x = self.up4(x, x2, t)
        x = self.sa8(x)
        x = self.up5(x, x1, t)
        x = self.sa9(x)
        output = self.outc(x)
        return output

class UNet_conditional_dc_5(nn.Module):
    def __init__(self, c_in=2+1, c_out=1, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.dc1 = DoubleConv(128, 128)
        self.down2 = Down(128, 256)
        self.dc2 = DoubleConv(256, 256)
        self.down3 = Down(256, 512)
        self.dc3 = DoubleConv(512, 512)
        self.down4 = Down(512, 1024)
        self.sa1 = SelfAttention(1024)
        self.down5 = Down(1024, 1024)
        self.sa2 = SelfAttention(1024)

        self.bot1 = DoubleConv(1024, 2048)
        self.bot2 = DoubleConv(2048, 2048)
        self.bot3 = DoubleConv(2048, 1024)

        self.up1 = Up(2048, 512)
        self.sa3 = SelfAttention(512)
        self.up2 = Up(1024, 256)
        self.sa4 = SelfAttention(256)
        self.up3 = Up(512, 128)
        self.dc4 = DoubleConv(128, 128)
        self.up4 = Up(256, 64)
        self.dc5 = DoubleConv(64, 64)
        self.up5 = Up(128, 64)
        self.dc6 = DoubleConv(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, c):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # if y is not None:
            # t += self.label_emb(y)
            # t += y
        # if self.conditional_method == "concat":
        #     t = torch.cat([t, y], dim=1)
        # else:
        #     t += y

        if c is not None:
            x0 = torch.cat([x, c], dim=1)
        else:
            y_channels = self.inc.double_conv[0].in_channels - x.shape[1]
            zero_padding = torch.zeros(x.shape[0], y_channels, x.shape[2], x.shape[3], device=self.device)
            x0 = torch.cat([x, zero_padding], dim=1)

        x1 = self.inc(x0)
        x2 = self.down1(x1, t)
        x2 = self.dc1(x2)
        x3 = self.down2(x2, t)
        x3 = self.dc2(x3)
        x4 = self.down3(x3, t)
        x4 = self.dc3(x4)
        x5 = self.down4(x4, t)
        x5 = self.sa1(x5)
        x6 = self.down5(x5, t)
        x6 = self.sa2(x6)

        x6 = self.bot1(x6)
        x6 = self.bot2(x6)
        x6 = self.bot3(x6)

        x = self.up1(x6, x5, t)
        x = self.sa3(x)
        x = self.up2(x, x4, t)
        x = self.sa4(x)
        x = self.up3(x, x3, t)
        x = self.dc4(x)
        x = self.up4(x, x2, t)
        x = self.dc5(x)
        x = self.up5(x, x1, t)
        x = self.dc6(x)
        output = self.outc(x)
        return output

if __name__ == '__main__':
    # net = UNet(device="cpu")
    net = UNet_conditional(num_classes=10, device="cpu")
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 1, 256, 256)
    t = x.new_tensor([500] * x.shape[0]).long()
    c = torch.randn(3, 2, 256, 256)
    print(net(x, t, c).shape)
