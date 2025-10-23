import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA
import logging
# from torch.utils.tensorboard import SummaryWriter
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


class Diffusion_ddim:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda", noise_add=True, cfg_scale=3):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device
        self.noise_add = noise_add
        self.cfg_scale = cfg_scale

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, num_inference_steps=50, noise_add=None, cfg_scale=None):
        if cfg_scale is not None:
            self.cfg_scale = cfg_scale
        if noise_add is not None:
            self.noise_add = noise_add
        # model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            labels = labels.to(self.device)
            timesteps = torch.linspace(self.noise_steps - 1, 0, steps=num_inference_steps, dtype=torch.long)
            # print(labels.shape)
            # for i in tqdm(range(num_inference_steps -1), position=0):
            for i in range(num_inference_steps -1):
                t = timesteps[i]
                prev_t = timesteps[i + 1]

                # Make t a tensor of shape (n,)
                t_tensor = (torch.ones(n) * t).long().to(self.device)

                # print(x.shape)
                predicted_noise = model(x, t_tensor, labels)
                if self.cfg_scale > 0:
                    uncond_predicted_noise = model(x, t_tensor.to(self.device), None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, self.cfg_scale)
                # alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t]
                alpha_hat_prev = self.alpha_hat[prev_t] if prev_t >= 0 else torch.tensor(1.0)  # Handle the last step

                # beta = self.beta[t][:, None, None, None]

                predicted_x0 = (x - torch.sqrt(1 - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat)
                predicted_x0 = torch.clamp(predicted_x0, -1.0, 1.0)
                x = (
                    torch.sqrt(alpha_hat_prev) * predicted_x0 +
                    torch.sqrt(1 - alpha_hat_prev) * predicted_noise
                )
                # if self.noise_add:
                #     if i > 1:
                #         noise = torch.randn_like(x)
                #     else:
                #         noise = torch.zeros_like(x)
                #     x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                # else:
                #     x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise)
        # model.train()
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        return x



class Diffusion_ddpm:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda", noise_add=True, cfg_scale=3):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device
        self.noise_add = noise_add
        self.cfg_scale = cfg_scale

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, num_inference_steps=50, noise_add=None, cfg_scale=None):
        if cfg_scale is not None:
            self.cfg_scale = cfg_scale
        if noise_add is not None:
            self.noise_add = noise_add
        # model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            labels = labels.to(self.device)
            # print(labels.shape)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                # print(x.shape)
                predicted_noise = model(x, t, labels)
                if self.cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, self.cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if self.noise_add:
                    if i > 1:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)
                        x = 1 / torch.sqrt(alpha) * (
                                    x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                            beta) * noise
                else:
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise)
            # model.train()
            # x = (x.clamp(-1, 1) + 1) / 2
            # x = (x * 255).type(torch.uint8)
            return x