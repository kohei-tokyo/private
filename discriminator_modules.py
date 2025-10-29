import torch.nn as nn
import timm


class Patch3(nn.Module):
    def __init__(self, in_channels=3):
        super(Patch3, self).__init__()

        def block(in_f, out_f, normalize=True):
            """Conv → (BN) → LeakyReLU"""
            layers = [nn.Conv2d(in_f, out_f, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, normalize=False),  # (N,64,H/2,W/2)
            *block(64, 128),  # (N,128,H/4,W/4)
            *block(128, 256),  # (N,256,H/8,W/8)
            # *block(256, 512),                          # (N,512,H/16,W/16)
            nn.Conv2d(256, 1, 4, padding=1)  # 出力 (N,1,H/16-1,W/16-1)
        )

    def forward(self, img):
        return self.model(img)  # "パッチごと" の真偽スコア


class Patch4(nn.Module):
    def __init__(self, in_channels=3):
        super(Patch4, self).__init__()

        def block(in_f, out_f, normalize=True):
            """Conv → (BN) → LeakyReLU"""
            layers = [nn.Conv2d(in_f, out_f, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, normalize=False),  # (N,64,H/2,W/2)
            *block(64, 128),  # (N,128,H/4,W/4)
            *block(128, 256),  # (N,256,H/8,W/8)
            *block(256, 512),  # (N,512,H/16,W/16)
            nn.Conv2d(512, 1, 4, padding=1)  # 出力 (N,1,H/16-1,W/16-1)
        )

    def forward(self, img):
        return self.model(img)  # "パッチごと" の真偽スコア


class Patch5(nn.Module):
    def __init__(self, in_channels=3):
        super(Patch5, self).__init__()

        def block(in_f, out_f, normalize=True):
            """Conv → (BN) → LeakyReLU"""
            layers = [nn.Conv2d(in_f, out_f, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, normalize=False),  # (N,64,H/2,W/2)
            *block(64, 128),  # (N,128,H/4,W/4)
            *block(128, 256),  # (N,256,H/8,W/8)
            *block(256, 512),  # (N,512,H/16,W/16)
            *block(512, 1024),  # (N,1024,H/32,W/32)
            nn.Conv2d(1024, 1, 4, padding=1)  # 出力 (N,1,H/32-1,W/32-1)
        )

    def forward(self, img):
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

    def forward(self, img):
        return self.model(img)
