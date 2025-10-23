import os
from tifffile import tifffile
from torch.utils.data import Dataset
import torch
import numpy as np
class DatasetDigitalStaining(Dataset):
    def __init__(self, folder, augmentation=None):
        self.img_path_list = os.listdir(folder)
        self.folder = folder
        self.augmentation = augmentation

    def __getitem__(self, i):
        image = tifffile.imread(os.path.join(self.folder, self.img_path_list[i]))
        phase1, phase2, mito = image[..., 0], image[..., 1], image[..., 2]

        if self.augmentation is not None:
            transformed = self.augmentation(image=phase1, image1=phase2, image2=mito)
            phase1, phase2, mito = transformed["image"], transformed["image1"], transformed["image2"]
            return phase1, phase2, torch.zeros_like(mito)
        else:
            phase1 = torch.from_numpy(phase1).unsqueeze(0)
            phase2 = torch.from_numpy(phase2).unsqueeze(0)
            mito = torch.from_numpy(mito).unsqueeze(0)
            return phase1, phase2, torch.zeros_like(mito)

    def __len__(self):
        return len(self.img_path_list)