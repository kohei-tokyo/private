import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from pytorch_msssim import ssim
import torch.nn as nn
import timm
from tqdm import tqdm
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import lpips
import itertools
import wandb


# %%
def tensor_ssim(img1, img2):
    return 1.0 - ssim(img1, img2, data_range=1.0, size_average=True)


def dice_loss_calc(pred, target, smooth=1):
    """
    Computes the Dice Loss for binary segmentation.
    Args:
        pred: Tensor of predictions (batch_size, 1, H, W).
        target: Tensor of ground truth (batch_size, 1, H, W).
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Scalar Dice Loss.
    """
    # Apply sigmoid to convert logits to probabilities
    # pred = torch.sigmoid(pred)
    # Calculate intersection and union
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    # Compute Dice Coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    # Return Dice Loss
    return 1 - dice.mean()


def lpips_calc(img1, img2, loss_fn):
    loss = loss_fn(img1, img2)
    return loss.mean()


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


class StainingGAN():
    def __init__(
            self,
            main_dir,
            train_folders=["1"],
            val_folders=["2"],
            test_folders=["3"],
            name="Run",
            n_epoch=50,
            discriminator="Patch4",  # Patch4, Patch3, Patch5, ResnetPatch, Resnet, or U_Net
            num_workers=8,  # GPUのメモリが足りない場合は小さくしてください
            in_chans=2,
            w_l1=50,
            w_ssim=1.0,
            w_dice=1.0,
            crop_size=256,
            stride=128,
            learning_rate_g=0.0002,
            learning_rate_d=0.0002,
            betas=(0.5, 0.999),
            images_to_use="both",
            device=torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'),
            patches_per_epoch=200,
            val_epoch=1,
            batch_size=16,
            *args, **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.device = device
        self.in_chans = in_chans
        self.n_epoch = n_epoch
        self.betas = betas
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.w_adv = 0.0 if discriminator == "U_Net" else 1.0
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_dice = w_dice
        self.crop_size = crop_size
        self.stride = stride
        self.lr_g = learning_rate_g
        self.lr_d = learning_rate_d
        self.l1_loss = torch.nn.L1Loss().to(device)
        self.loss_fn_mse = torch.nn.MSELoss().to(device)
        self.loss_fn_ssim = tensor_ssim
        self.loss_fn_lpips = lpips.LPIPS(net='alex').to(device)
        self.last_batch_with_pred = None
        self.name = name
        self.images_to_use = images_to_use
        self.patches_per_epoch = patches_per_epoch
        self.val_epoch = val_epoch
        self.class_num = 1
        self.hist = {"train": [], "val": [], "test": []}
        self.min_val_loss_mse = 100000
        self.min_val_loss_ssim = 100000
        self.min_val_loss_lpips = 100000
        self.min_val_loss_dice_loss = 100000
        self.test_id = ""

        self.G = smp.Unet(
            encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_chans,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=self.class_num,  # model output channels (number of classes in your dataset)
        ).to(device)
        if discriminator == "Patch3":
            self.D = Patch3(in_channels=in_chans + self.class_num).to(device)
        elif discriminator == "Patch4":
            self.D = Patch4(in_channels=in_chans + self.class_num).to(device)
        elif discriminator == "Patch5":
            self.D = Patch5(in_channels=in_chans + self.class_num).to(device)
        elif discriminator == "ResnetPatch":
            self.D = ResnetPatch(in_channels=in_chans + self.class_num).to(device)
        else:
            self.D = timm.create_model("resnet18",
                                       in_chans=in_chans + self.class_num,
                                       pretrained=False,
                                       num_classes=1).to(device)
        self.optimizer_g = torch.optim.Adam(self.G.parameters(), lr=self.lr_g, betas=self.betas)
        self.optimizer_d = torch.optim.Adam(self.D.parameters(), lr=self.lr_d, betas=self.betas)

        from make_dataset import DatasetDigitalStaining
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        img_folders = [os.path.join(main_dir, f) for f in train_folders]
        datasets = [DatasetDigitalStaining(img_folders[i], augmentation=None) for i in range(len(train_folders))]
        combined_dataset = ConcatDataset(datasets)
        self.train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                       pin_memory=True, persistent_workers=True)
        img_folders = [os.path.join(main_dir, f) for f in val_folders]
        datasets = [DatasetDigitalStaining(img_folders[i], augmentation=None) for i in range(len(val_folders))]
        combined_dataset = ConcatDataset(datasets)
        self.val_loader = DataLoader(combined_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
        img_folders = [os.path.join(main_dir, f) for f in test_folders]
        datasets = [DatasetDigitalStaining(img_folders[i], augmentation=None) for i in range(len(test_folders))]
        combined_dataset = ConcatDataset(datasets)
        self.test_loader = DataLoader(combined_dataset, batch_size=8, shuffle=False)


    def _wandb_init(self):
        self.run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="kohei_tokyo-the-university-of-tokyo",
            # Set the wandb project where this run will be logged.
            project="Digital_Staining",
            name=self.name,
            # Track hyperparameters and run metadata.
            config={
            },
        )

    def train(self):
        self._wandb_init()
        for self.epoch in range(self.n_epoch):
            print(f"Epoch {self.epoch + 1}/{self.n_epoch}")
            self.calc_epoch("train")
            if self.epoch % self.val_epoch == 0:
                self.calc_epoch("val")
                self.show_result("val")
        print(f"min_epoch_mse {self.min_epoch_mse}")
        print(f"min_epoch_ssim {self.min_epoch_ssim}")
        print(f"min_epoch_lpips {self.min_epoch_lpips}")

    def test(self):
        test_list = ["ssim", "mse", "lpips"]
        for self.test_id in test_list:
            print(f"Test {self.test_id}")
            if self.test_id != "final":
                self.G.load_state_dict(torch.load(f"path//best_model_G_stain_{self.test_id}_{self.name}.pth"))
                self.D.load_state_dict(torch.load(f"path//best_model_D_stain_{self.test_id}_{self.name}.pth"))
                self.G.to(self.device)
                self.D.to(self.device)
            self.calc_epoch("test")
            self.show_result("test")
        wandb.finish()

    def calc_epoch(self, mode):
        if mode == "train":
            self.G.train()
            self.D.train()
            loader = itertools.islice(self.train_loader, self.patches_per_epoch)
            patches_num = self.patches_per_epoch
            grad_ctx = torch.enable_grad()
        elif mode == "val":
            self.G.eval()
            self.D.eval()
            loader = self.val_loader
            patches_num = len(loader)
            grad_ctx = torch.no_grad()
        elif mode == "test":
            self.G.eval()
            self.D.eval()
            loader = self.test_loader
            patches_num = len(loader)
            grad_ctx = torch.no_grad()
        else:
            raise NotImplementedError
        total_metrics_dict = None

        with grad_ctx:
            for ph1, ph2, real, real_mask in tqdm(loader, total=patches_num):
                metrics = self.calc_batch(ph1, ph2, real, real_mask, mode)
                if total_metrics_dict is None:
                    total_metrics_dict = {k: 0 for k, v in metrics.items()}
                for k, v in metrics.items():
                    total_metrics_dict[k] += metrics[k].item()

        for k, v in total_metrics_dict.items():
            total_metrics_dict[k] /= patches_num
        self.save_results(total_metrics_dict, mode)

    def calc_batch(self, ph1, ph2, real, real_mask, mode):
        return self.calc_matrix(ph1, ph2, real, real_mask, mode)
        # if mode == "train":
        #     return self.calc_matrix(ph1, ph2, real, real_mask, mode)
        # else:
        #     metrics_dict = None
        #     _, _, H, W = ph1.shape
        #     len_met = 0
        #     for i in range(0, H - self.crop_size + 1, self.stride):
        #         for j in range(0, W - self.crop_size + 1, self.stride):
        #             len_met += 1
        #             metrics = self.calc_matrix(
        #                 # ph1,
        #                 # ph2,
        #                 # real,
        #                 # real_mask,
        #                 ph1[:, :, i:i + self.crop_size, j:j + self.crop_size],
        #                 ph2[:, :, i:i + self.crop_size, j:j + self.crop_size],
        #                 real[:, :, i:i + self.crop_size, j:j + self.crop_size],
        #                 real_mask[:, i:i + self.crop_size, j:j + self.crop_size],
        #                 mode
        #             )
        #             if metrics_dict is None:
        #                 metrics_dict = {k: 0 for k, v in metrics.items()}
        #             for k, v in metrics.items():
        #                 metrics_dict[k] += metrics[k]
        #
        #     for k, v in metrics_dict.items():
        #         metrics_dict[k] /= len_met
        #     return metrics_dict

    def calc_matrix(self, ph1, ph2, real, real_mask, mode):
        x = torch.concat([ph1, ph2], dim=1).to(self.device)
        real = real.to(self.device)
        # real_mask = real_mask.unsqueeze(1).to(self.device)
        # fake, fake_mask = self.G(x).split(1, dim=1)
        fake = F.sigmoid(self.G(x))
        # real_pair = torch.cat([x, real, real_mask], dim=1)
        # fake_pair = torch.cat([x, fake, fake_mask], dim=1)
        real_pair = torch.cat([x, real], dim=1)
        fake_pair = torch.cat([x, fake], dim=1)

        loss_d, acc_real, acc_fake = self.calc_dis(real_pair, fake_pair, mode)
        # adv_loss, l1_loss, loss_g, mse, ssim_loss, dice_loss = self.calc_gen(
        #     real_pair, fake_pair, mode, real, real_mask, fake, fake_mask
        # )
        adv_loss, loss_g, l1_loss, ssim_loss, mse, lpips_loss = self.calc_gen(
            real_pair, fake_pair, mode, real, fake
        )

        return {
            "adv_loss_g": adv_loss, "l1_loss": l1_loss, "loss_g": loss_g, "loss_d": loss_d,
            "mse": mse, "ssim": ssim_loss, "lpips": lpips_loss, "acc_real": acc_real, "acc_fake": acc_fake
        }

    def calc_dis(self, real_pair, fake_pair, mode):
        pred_real = self.D(real_pair)
        pred_fake = self.D(fake_pair.detach())
        target_real = torch.ones_like(pred_real).to(self.device)
        target_fake = torch.zeros_like(pred_fake).to(self.device)

        loss_fake = self.bce_loss(pred_fake, target_fake)
        loss_real = self.bce_loss(pred_real, target_real)
        loss_d = (loss_real + loss_fake) * 0.5
        acc_real = pred_real.sigmoid().float().mean()
        acc_fake = pred_fake.sigmoid().float().mean()

        if mode == "train":
            self.optimizer_d.zero_grad()
            loss_d.backward()
            self.optimizer_d.step()

        return loss_d, acc_real, acc_fake

    def calc_gen(self, real_pair, fake_pair, mode, real, fake):
        pred_fake = self.D(fake_pair)
        target_fake = torch.ones_like(pred_fake).to(self.device)

        adv_loss = self.bce_loss(pred_fake, target_fake)
        l1_loss = self.l1_loss(real, fake)
        ssim_loss = self.loss_fn_ssim(real, fake)
        mse = self.loss_fn_mse(real, fake)
        if mode == "train":
            lpips_loss = torch.tensor(0.0)
        else:
            lpips_loss = lpips_calc(real, fake, self.loss_fn_lpips)
        # dice_loss = dice_loss_calc(fake_mask, real_mask)
        # loss_g = adv_loss + self.w_l1 * l1_loss + self.w_ssim * ssim_loss + self.w_dice * dice_loss
        loss_g = self.w_adv * adv_loss + self.w_l1 * l1_loss + self.w_ssim * ssim_loss

        if mode == "train":
            self.optimizer_g.zero_grad()
            loss_g.backward()
            self.optimizer_g.step()

        return adv_loss, loss_g, l1_loss, ssim_loss, mse, lpips_loss

    def save_results(self, total_metrics_dict, mode):
        self.hist[mode].append(total_metrics_dict)
        for k, v in total_metrics_dict.items():
            print(f"{mode} {k}: {v}")
        total_metrics_dict_log = {f"{mode}_"+k:v for k,v in total_metrics_dict.items()}
        self.run.log(total_metrics_dict_log)

        if mode == "val":
            mean_mse = total_metrics_dict['mse']
            mean_ssim = total_metrics_dict['ssim']
            mean_lpips = total_metrics_dict['lpips']
            if mean_mse < self.min_val_loss_mse:
                print(f"Loss_mse improved to {mean_mse}, saving model")
                self.min_val_loss_mse = mean_mse
                self.min_epoch_mse = self.epoch
                torch.save(self.G.state_dict(), f"path//best_model_G_stain_mse_{self.name}.pth")
                torch.save(self.D.state_dict(), f"path//best_model_D_stain_mse_{self.name}.pth")
            if mean_ssim < self.min_val_loss_ssim:
                self.min_val_loss_ssim = mean_ssim
                self.min_epoch_ssim = self.epoch
                print(f"Loss_ssim improved to {mean_ssim}, saving model")
                torch.save(self.G.state_dict(), f"path//best_model_G_stain_ssim_{self.name}.pth")
                torch.save(self.D.state_dict(), f"path//best_model_D_stain_ssim_{self.name}.pth")
            if mean_lpips < self.min_val_loss_lpips:
                self.min_val_loss_lpips = mean_lpips
                self.min_epoch_lpips = self.epoch
                print(f"Loss_lpips improved to {mean_lpips}, saving model")
                torch.save(self.G.state_dict(), f"path//best_model_G_stain_lpips_{self.name}.pth")
                torch.save(self.D.state_dict(), f"path//best_model_D_stain_lpips_{self.name}.pth")

    def show_result(self, mode):
        if mode == "val":
            loader = self.val_loader
            n = 1
        elif mode == "test":
            loader = self.test_loader
            n = 6
        else:
            loader = self.train_loader
            n = 1

        for ph1, ph2, real, real_mask in loader:
            _, _, H, W = ph1.shape
            # output_pred = torch.zeros(1, 2, H, W).to(self.device)
            # count_map = torch.zeros(1, 1, H, W).to(self.device)
            #
            # for i in range(0, H - self.crop_size + 1, self.stride):
            #     for j in range(0, W - self.crop_size + 1, self.stride):
            #         ph1_crop = ph1[:, :, i:i + self.crop_size, j:j + self.crop_size].to(self.device)
            #         ph2_crop = ph2[:, :, i:i + self.crop_size, j:j + self.crop_size].to(self.device)
            #         x_crop = torch.cat([ph1_crop, ph2_crop], dim=1)
            #
            #         with torch.no_grad():
            #             pred_crop = self.G(x_crop)  # [1,2,crop_size,crop_size]
            #
            #         output_pred[:, :, i:i + self.crop_size, j:j + self.crop_size] += pred_crop[0]
            #         count_map[:, :, i:i + self.crop_size, j:j + self.crop_size] += 1
            #
            # output_pred /= count_map
            # fake = F.sigmoid(output_pred[:, 0][0].squeeze()).cpu().detach().numpy()

            x_crop = torch.cat([ph1.to(self.device), ph2.to(self.device)], dim=1)
            with torch.no_grad():
                pred_crop = self.G(x_crop)  # [1,2,crop_size,crop_size]
            output_pred = pred_crop[0][0]
            fake = F.sigmoid(output_pred).cpu().detach().numpy()

            # fake_mask = F.sigmoid(output_pred[:,0][0].squeeze()).cpu().detach().numpy()

            # x = torch.concat([ph1, ph2], dim=1).to(self.device)
            # prediction = self.G(x)
            # fake = prediction[:,0][0].squeeze().cpu().detach().numpy()
            # fake_mask = prediction[:,1][0].squeeze().cpu().detach().numpy()
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            # axs[0,0].imshow(real[0].squeeze())
            # axs[0,0].axis('off')
            # axs[0,0].set_title('target')
            # axs[0,1].imshow(fake)
            # axs[0,1].axis('off')
            # axs[0,1].set_title('target_pred')
            axs[0].imshow(real[0].squeeze())
            axs[0].axis('off')
            axs[0].set_title('target')
            axs[1].imshow(fake)
            axs[1].axis('off')
            axs[1].set_title('prediction')
            plt.tight_layout()
            plt.show()
            n = n - 1
            # if self.epoch % 5 == 0:
            wandb.log({f"{mode}_{self.test_id}_pred" : wandb.Image(fake * 255.0)})
            if n <= 0:
                break