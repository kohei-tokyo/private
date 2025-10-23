import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, UNet_conditional_ori, UNet_conditional_deep, UNet_conditional_sa_5, UNet_conditional_dc_5, EMA
from modules_2 import UNet_gemini
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from pytorch_msssim import ssim
import timm
import torch.nn.functional as F
import lpips
import itertools
import wandb
from make_dataset import DatasetDigitalStaining
from ddpm_conditional import Diffusion_ddim, Diffusion_ddpm


def tensor_ssim(img1, img2):
    return 1.0 - ssim(img1, img2, data_range=1.0, size_average=True)

class DDPM(nn.Module):
    def __init__(
            self,
            dir,
            name="Run",  # 名称
            group=None,
            train_folders=["train"],  # Trainデータのフォルダ名
            val_folders=["val"],  # Valデータのフォルダ名
            test_folders=["test"],  # Testデータのフォルダ名
            n_epoch=10,
            num_workers=8,  # GPUのメモリが足りない場合は小さくしてください
            in_chans=2,
            learning_rate=3e-4,
            images_to_use="both",
            device=torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'),
            patches_per_epoch=200,
            patches_per_epoch_val=5,
            val_epoch=1,
            batch_size=16,
            image_size=256,
            noise_add=True,
            cfg_scale=3,
            no_label=True,
            noise_steps=10,
            num_inference_steps=50,
            mode_dif="ddim",
            model_unet="original",
            dim_mults=(1, 2, 4, 4),
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.test_id = None
        self.dir = dir
        self.name = name
        self.group = group
        self.train_folders = train_folders
        self.val_folders = val_folders
        self.test_folders = test_folders
        self.n_epoch = n_epoch
        self.num_workers = num_workers
        self.in_chans = in_chans
        self.learning_rate = learning_rate
        self.images_to_use = images_to_use
        self.device = device
        self.patches_per_epoch = patches_per_epoch
        self.patches_per_epoch_val = patches_per_epoch_val
        self.val_epoch = val_epoch
        self.batch_size = batch_size
        self.image_size = image_size
        self.noise_add = noise_add
        self.cfg_scale = cfg_scale
        self.no_label = no_label
        self.noise_steps = noise_steps
        self.num_inference_steps = num_inference_steps
        self.mode_dif = mode_dif

        if model_unet == "original":
            self.model = UNet_conditional_ori(
                c_in=in_chans + 1,
                c_out=1,
                device=device
            ).to(device)
        elif model_unet == "deep":
            self.model = UNet_conditional_deep(
                c_in=in_chans + 1,
                c_out=1,
                device=device
            ).to(device)
        elif model_unet == "dc_5":
            self.model = UNet_conditional_dc_5(
                c_in=in_chans + 1,
                c_out=1,
                device=device
            ).to(device)
        elif model_unet == "sa_5":
            self.model = UNet_conditional_sa_5(
                c_in=in_chans + 1,
                c_out=1,
                device=device
            ).to(device)
        elif model_unet == "gemini":
            self.model = UNet_gemini(
                c_in=in_chans + 1,
                c_out=1,
                dim_mults=dim_mults,
                device=device
            ).to(device)
        else:
            self.model = UNet_conditional(
                c_in=in_chans + 1,
                c_out=1,
                device=device
            ).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        if mode_dif == "ddim":
            self.diffusion = Diffusion_ddim(noise_steps=noise_steps, img_size=image_size, device=device,
                                            noise_add=noise_add, cfg_scale=cfg_scale)
        else:
            self.diffusion = Diffusion_ddpm(noise_steps=noise_steps, img_size=image_size, device=device,
                                            noise_add=noise_add, cfg_scale=cfg_scale)
        self.ema = EMA(0.995)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.loss_fn_mse = nn.MSELoss().to(device)
        self.loss_fn_ssim = tensor_ssim
        self.loss_fn_lpips = lpips.LPIPS(net='alex').to(device)
        self.hist = {"train": [], "val": [], "test": []}
        self.val_list = [
            "mse",
            "ssim",
            "lpips",
            "mse_ema",
            "ssim_ema",
            "lpips_ema",
        ]
        self.best_score_list = [float('inf')] * len(self.val_list)
        self.min_epoch_list = [0] * len(self.val_list)
        self.epoch = 0

        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        img_folders = [os.path.join(dir, f) for f in train_folders]
        datasets = [DatasetDigitalStaining(img_folders[i], augmentation=None) for i in range(len(train_folders))]
        combined_dataset = ConcatDataset(datasets)
        self.train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                       pin_memory=True, persistent_workers= num_workers > 0)
        img_folders = [os.path.join(dir, f) for f in val_folders]
        datasets = [DatasetDigitalStaining(img_folders[i], augmentation=None) for i in range(len(val_folders))]
        combined_dataset = ConcatDataset(datasets)
        self.val_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                     pin_memory=True, persistent_workers= num_workers > 0)
        img_folders = [os.path.join(dir, f) for f in test_folders]
        datasets = [DatasetDigitalStaining(img_folders[i], augmentation=None) for i in range(len(test_folders))]
        combined_dataset = ConcatDataset(datasets)
        self.test_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                     pin_memory=True, persistent_workers= num_workers > 0)

    def all(self):
        self.train()
        self.test()

    def _wandb_init(self):
        self.run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="kohei_tokyo-the-university-of-tokyo",
            # Set the wandb project where this run will be logged.
            project="Digital_Staining",
            name=self.name,
            group=self.group,
            # Track hyperparameters and run metadata.
            config={},
        )

    def train(self, noise_add=None, cfg_scale=None, mode_dif=None):
        if cfg_scale is not None:
            self.cfg_scale = cfg_scale
        if noise_add is not None:
            self.noise_add = noise_add
        if mode_dif is not None:
            self.mode_dif = mode_dif
        self._wandb_init()
        for self.epoch in range(self.n_epoch):
            print(f"Epoch {self.epoch + 1}/{self.n_epoch}")
            self.calc_epoch("train")
            torch.save(self.model.state_dict(), f"path//best_model_stain_final_{self.name}.pth")
            torch.save(self.ema_model.state_dict(), f"path//best_model_stain_final_ema_{self.name}.pth")
            if self.epoch % self.val_epoch == 0:
                self.calc_epoch("val")
                self.show_result("val")

        for val_n in range(len(self.val_list)):
            print(f"min_epoch_{self.val_list[val_n]} {self.min_epoch_list[val_n] + 1}")

    def test(self, noise_add=None, cfg_scale=None, mode_dif=None):
        if cfg_scale is not None:
            self.cfg_scale = cfg_scale
        if noise_add is not None:
            self.noise_add = noise_add
        if mode_dif is not None:
            self.mode_dif = mode_dif
        test_list = ["final", "ssim", "mse", "lpips"]
        for self.test_id in test_list:
            print(f"Test {self.test_id}")
            self.model.load_state_dict(torch.load(f"path//best_model_stain_{self.test_id}_{self.name}.pth"))
            self.model.to(self.device)
            self.calc_epoch("test")
            self.show_result("test")
        wandb.finish()

    def calc_epoch(self, mode):
        if mode == "train":
            self.model.train()
            loader = itertools.islice(self.train_loader, self.patches_per_epoch)
            patches_num = self.patches_per_epoch
            grad_ctx = torch.enable_grad()
        elif mode == "val":
            self.model.eval()
            loader = itertools.islice(self.val_loader, self.patches_per_epoch_val)
            # loader = self.val_loader
            patches_num = self.patches_per_epoch_val
            grad_ctx = torch.no_grad()
        elif mode == "test":
            self.model.eval()
            loader = itertools.islice(self.test_loader, self.patches_per_epoch_val)
            # loader = self.test_loader
            patches_num = self.patches_per_epoch_val
            grad_ctx = torch.no_grad()
        else:
            raise NotImplementedError
        total_metrics_dict = None

        with grad_ctx:
            for ph1, ph2, real in tqdm(loader, total=patches_num):
                metrics = self.calc_batch(ph1, ph2, real, mode)
                # print(metrics)
                if total_metrics_dict is None:
                    total_metrics_dict = {k: 0 for k, v in metrics.items()}
                for k, v in metrics.items():
                    total_metrics_dict[k] += metrics[k].item()

        for k, v in total_metrics_dict.items():
            total_metrics_dict[k] /= patches_num
        self.save_results(total_metrics_dict, mode)

    def calc_batch(self, ph1, ph2, real, mode):
        if mode == "train":
            return self.calc_matrix_train(ph1, ph2, real)
        else:
            return self.calc_matrix_test(ph1, ph2, real)

    def calc_matrix_train(self, ph1, ph2, real):
        x = torch.concat([ph1, ph2], dim=1).to(self.device)
        real = real.to(self.device)
        # predict = F.sigmoid(self.model(x))

        t = self.diffusion.sample_timesteps(real.shape[0]).to(self.device)
        x_t, noise = self.diffusion.noise_images(real, t)
        if self.no_label:
            if np.random.random() < 0.1:
                x = None
        predicted_noise = self.model(x_t, t, x)
        loss = self.loss_fn_mse(noise, predicted_noise)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.ema.step_ema(self.ema_model, self.model)
        return {"mse": loss}

    def calc_matrix_test(self, ph1, ph2, real):
        x = torch.concat([ph1, ph2], dim=1).to(self.device)
        real = real.to(self.device)

        ema_sampled_images = self.diffusion.sample(self.ema_model, n=self.batch_size, labels=x, noise_add=self.noise_add, cfg_scale=self.cfg_scale)
        mse_loss_ema = self.loss_fn_mse(real, ema_sampled_images)
        ssim_loss_ema = self.loss_fn_ssim(real, ema_sampled_images)
        lpips_loss_ema = self.loss_fn_lpips(real, ema_sampled_images).mean()

        sampled_images = self.diffusion.sample(self.model, n=self.batch_size, labels=x, num_inference_steps=self.num_inference_steps, noise_add=self.noise_add, cfg_scale=self.cfg_scale)
        # print(real.shape, sampled_images.shape, ema_sampled_images.shape)
        mse_loss = self.loss_fn_mse(real, sampled_images)
        ssim_loss = self.loss_fn_ssim(real, sampled_images)
        lpips_loss = self.loss_fn_lpips(real, sampled_images).mean()

        return {
            "mse": mse_loss,
            "mse_ema": mse_loss_ema,
            "ssim": ssim_loss,
            "ssim_ema": ssim_loss_ema,
            "lpips": lpips_loss,
            "lpips_ema": lpips_loss_ema,
        }

    def save_results(self, total_metrics_dict, mode):
        self.hist[mode].append(total_metrics_dict)
        for k, v in total_metrics_dict.items():
            print(f"{mode} {k}: {v}")
        total_metrics_dict_log = {f"{mode}_" + k: v for k, v in total_metrics_dict.items()}
        self.run.log(total_metrics_dict_log)

        if mode == "val":
            torch.save(self.model.state_dict(), f"path//best_model_stain_final_{self.name}.pth")
            torch.save(self.ema_model.state_dict(), f"path//best_model_stain_final_ema_{self.name}.pth")
            for val_n in range(len(self.val_list)):
                mean_loss = total_metrics_dict[self.val_list[val_n]]
                if mean_loss < self.best_score_list[val_n]:
                    print(f"Loss_{self.val_list[val_n]} improved to {mean_loss}, saving model")
                    self.best_score_list[val_n] = mean_loss
                    self.min_epoch_list[val_n] = self.epoch
                    torch.save(self.model.state_dict(),f"path//best_model_stain_{self.val_list[val_n]}_{self.name}.pth")
                    if val_n < len(self.val_list) / 2:
                        torch.save(self.model.state_dict(), f"path//best_model_stain_{self.val_list[val_n]}_{self.name}.pth")
                    else:
                        torch.save(self.ema_model.state_dict(), f"path//best_model_stain_{self.val_list[val_n]}_{self.name}.pth")

    def show_result(self, mode, image_n=1):
        if mode == "val":
            loader = self.val_loader
            n = image_n
        elif mode == "test":
            loader = self.test_loader
            n = 6
        else:
            loader = self.train_loader
            n = 1

        for ph1, ph2, real in loader:
            if n <= 1:
                _, _, H, W = ph1.shape
                x = torch.cat([ph1[0:1].to(self.device), ph2[0:1].to(self.device)], dim=1)
                with torch.no_grad():
                    pred = self.diffusion.sample(self.model, n=1, labels=x, num_inference_steps=self.num_inference_steps,
                                                 noise_add=self.noise_add, cfg_scale=self.cfg_scale)
                    pred_ema = self.diffusion.sample(self.ema_model, n=1, labels=x, num_inference_steps=self.num_inference_steps,
                                                 noise_add=self.noise_add, cfg_scale=self.cfg_scale)
                output_pred = pred[0][0].cpu().detach().numpy()
                output_pred_ema = pred_ema[0][0].cpu().detach().numpy()
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(real[0].squeeze())
                axs[0].axis('off')
                axs[0].set_title('target')
                axs[1].imshow(output_pred)
                axs[1].axis('off')
                axs[1].set_title('prediction')
                axs[2].imshow(output_pred_ema)
                axs[2].axis('off')
                axs[2].set_title('ema_prediction')
                plt.tight_layout()
                plt.show()
                wandb.log({f"{mode}_{self.test_id}_pred": wandb.Image(((np.clip(output_pred, -1, 1) + 1) / 2) * 255)})
                wandb.log({f"{mode}_{self.test_id}_pred_ema": wandb.Image(((np.clip(output_pred_ema, -1, 1) + 1) / 2) * 255)})
            n = n - 1
            if mode == "val":
                if n <= 0:
                    break
            else:
                if n <= -5:
                    break

    def result_image(self, n_first=1, n_long=1, num_inference_steps=50, noise_add=True, cfg_scale=3, model="normal"):
        loader = self.val_loader
        n = n_first + n_long - 1
        for ph1, ph2, real in loader:
            if n <= n_long:
                _, _, H, W = ph1.shape
                x = torch.cat([ph1[0:1].to(self.device), ph2[0:1].to(self.device)], dim=1)
                with torch.no_grad():
                    if model == "normal":
                        pred = self.diffusion.sample(self.model, n=1, labels=x, num_inference_steps=num_inference_steps,
                                                     noise_add=noise_add, cfg_scale=cfg_scale)
                    else:
                        pred = self.diffusion.sample(self.ema_model, n=1, labels=x, num_inference_steps=num_inference_steps,
                                                     noise_add=noise_add, cfg_scale=cfg_scale)
                output_pred = pred[0][0].cpu().detach().numpy()
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].imshow(real[0].squeeze())
                axs[0].axis('off')
                axs[0].set_title('target')
                axs[1].imshow(output_pred)
                axs[1].axis('off')
                axs[1].set_title('prediction')
                plt.tight_layout()
                plt.show()
            n = n - 1
            if n <= 0:
                break


# def _train(args):
#     setup_logging(args.run_name)
#     device = args.device
#     dataloader = get_data(args)
#     model = UNet_conditional(num_classes=args.num_classes).to(device)
#     optimizer = optim.AdamW(model.parameters(), lr=args.lr)
#     mse = nn.MSELoss()
#     diffusion = Diffusion(img_size=args.image_size, device=device)
#     logger = SummaryWriter(os.path.join("runs", args.run_name))
#     l = len(dataloader)
#     ema = EMA(0.995)
#     ema_model = copy.deepcopy(model).eval().requires_grad_(False)
#
#     for epoch in range(args.epochs):
#         logging.info(f"Starting epoch {epoch}:")
#         pbar = tqdm(dataloader)
#         for i, (images, labels) in enumerate(pbar):
#             images = images.to(device)
#             labels = labels.to(device)
#             t = diffusion.sample_timesteps(images.shape[0]).to(device)
#             x_t, noise = diffusion.noise_images(images, t)
#             if np.random.random() < 0.1:
#                 labels = None
#             predicted_noise = model(x_t, t, labels)
#             loss = mse(noise, predicted_noise)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             ema.step_ema(ema_model, model)
#
#             pbar.set_postfix(MSE=loss.item())
#             logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
#
#         if epoch % 10 == 0:
#             labels = torch.arange(10).long().to(device)
#             sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
#             ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
#             plot_images(sampled_images)
#             save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
#             save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
#             torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
#             torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
#             torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))
#
#
# def launch():
#     import argparse
#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()
#     args.run_name = "DDPM_conditional"
#     args.epochs = 300
#     args.batch_size = 14
#     args.image_size = 64
#     args.num_classes = 10
#     args.dataset_path = r"C:\Users\dome\datasets\cifar10\cifar10-64\train"
#     args.device = "cuda"
#     args.lr = 3e-4
#     train(args)
#
#
# if __name__ == '__main__':
#     launch()
#     # device = "cuda"
#     # model = UNet_conditional(num_classes=10).to(device)
#     # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
#     # model.load_state_dict(ckpt)
#     # diffusion = Diffusion(img_size=64, device=device)
#     # n = 8
#     # y = torch.Tensor([6] * n).long().to(device)
#     # x = diffusion.sample(model, n, y, cfg_scale=0)
#     # plot_images(x)

