import os
from tifffile import tifffile
from torch.utils.data import Dataset
import torch
import numpy as np
class DatasetDigitalStaining(Dataset):
    def __init__(self, folder, prompt, tokenizer, RGB_channels=False, augmentation=None):
        self.tokenizer = tokenizer
        self.img_path_list = os.listdir(folder)
        self.folder = folder
        self.prompt = prompt
        self.RGB_channels = RGB_channels
        self.augmentation = augmentation

    def __getitem__(self, i):
        image = tifffile.imread(os.path.join(self.folder, self.img_path_list[i]))
        phase1, phase2, mito = image[..., 0], image[..., 1], image[..., 2]

        if self.augmentation is not None:
            transformed = self.augmentation(image=phase1, image1=phase2, image2=mito)
            phase1, phase2, mito = transformed["image"], transformed["image1"], transformed["image2"]
        else:
            phase1 = torch.from_numpy(phase1).unsqueeze(0)
            phase2 = torch.from_numpy(phase2).unsqueeze(0)
            mito = torch.from_numpy(mito).unsqueeze(0)

        target_image = mito
        conditioning_image = torch.concat([phase1, phase2], dim=0)

        if self.RGB_channels:
            zero_image = torch.zeros_like(mito)
            target_image = torch.cat([target_image, zero_image, zero_image], dim=0)
            conditioning_image = torch.cat([conditioning_image, zero_image], dim=0)

        input_ids = self.tokenizer(
            self.prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        return {
            "target_image": target_image,
            "conditioning_image": target_image,
            # "conditioning_image": conditioning_image,
            "input_ids": input_ids.squeeze(0),
            "prompt": self.prompt
        }

    def __len__(self):
        return len(self.img_path_list)