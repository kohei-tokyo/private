import os
from tifffile import tifffile
import numpy as np
import albumentations as A
import cv2
from tqdm import tqdm


def normalize_clip(img, vmin, vmax):
    img = np.clip(img, vmin, vmax)
    return (img - vmin) / (vmax - vmin)


def standardization(img):
    return (img - img.mean()) / img.std()


def stand_norm(img, img_minmax):
    img = standardization(img)
    return normalize_clip(img, img_minmax[0], img_minmax[1])


def produce_images(
        original_dir,
        main_dir,
        train_folders=["1"],
        val_folders=["2"],
        test_folders=["3"],
        img_n=100,
        img_size=256
):
    print("produce_images start")
    original_img_folders = [os.path.join(original_dir, f) for f in train_folders]
    img_list = []
    for i in range(len(original_img_folders)):
        img_path_list = os.listdir(original_img_folders[i])
        for j in range(len(img_path_list)):
            image = tifffile.imread(os.path.join(original_img_folders[i], img_path_list[j]))
            img = []
            for l in range(len(image[0][0])):
                img.append(standardization(image[..., l]).flatten())
            img_list.append([img])
    img_list = np.array(img_list)
    img_minmax = []
    for l in range(len(image[0][0])):
        img_concat = np.concatenate(img_list[:, 0, l, :])
        img_minmax.append([np.percentile(img_concat, 0.1), np.percentile(img_concat, 99.9)])
    print("norm_measure done")

    train_aug = A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.2, scale_limit=0.2,
                rotate_limit=30, p=0.7,
                border_mode=cv2.BORDER_REFLECT_101
            ),
            A.CropNonEmptyMaskIfExists(
                height=img_size,
                width=img_size,
                p=1.0
            ),
        ],
        additional_targets={"image1": "image",
                            "image2": "image",
                            "image3": "image"},
        strict=True,
        seed=137,
    )
    original_img_folders = [os.path.join(original_dir, f) for f in train_folders]
    img_folders = [os.path.join(main_dir, f) for f in train_folders]
    for folder_path in img_folders:
        os.makedirs(folder_path, exist_ok=True)
    for i in range(len(original_img_folders)):
        img_path_list = os.listdir(original_img_folders[i])
        for j in tqdm(range(len(img_path_list))):
            image = tifffile.imread(os.path.join(original_img_folders[i], img_path_list[j]))
            for l in range(len(image[0][0])):
                image[..., l] = stand_norm(image[..., l], img_minmax[l])
            phase1, phase2, mito = image[..., 0], image[..., 1], image[..., 2]
            mask = (mito > 0).astype(np.float32)
            for k in range(img_n):
                augmented = train_aug(image=phase1, image1=phase2, image2=mito, mask=mask)
                image_crop = np.stack([augmented['image'], augmented['image1'], augmented['image2']], axis=-1)
                base_name, ext = os.path.splitext(img_path_list[j])
                new_filename = f"{base_name}_{k}{ext}"
                save_path = os.path.join(img_folders[i], new_filename)
                tifffile.imwrite(save_path, image_crop)
        print(f"train_folder {img_folders[i]} done")

    original_img_folders = [os.path.join(original_dir, f) for f in (val_folders + test_folders)]
    img_folders = [os.path.join(main_dir, f) for f in (val_folders + test_folders)]
    for folder_path in img_folders:
        os.makedirs(folder_path, exist_ok=True)
    for i in range(len(original_img_folders)):
        img_path_list = os.listdir(original_img_folders[i])
        for j in range(len(img_path_list)):
            image = tifffile.imread(os.path.join(original_img_folders[i], img_path_list[j]))
            for l in range(len(image[0][0])):
                image[..., l] = stand_norm(image[..., l], img_minmax[l])
            tifffile.imwrite(os.path.join(img_folders[i], img_path_list[j]), image)
        print(f"val_test_folder {img_folders[i]} done")
    print("produce_images done")