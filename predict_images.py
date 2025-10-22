import os
from tifffile import tifffile
import torch
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp
import torch.nn.functional as F
#%%
def normalize_clip(img, vmin, vmax):
    img = np.clip(img, vmin, vmax)
    return (img - vmin) / (vmax - vmin)

def standardization(img):
    return (img - img.mean()) / img.std()

def stand_norm(img, img_minmax):
    img = standardization(img)
    return normalize_clip(img, img_minmax[0], img_minmax[1])

#%%
def PredictGAN(
        dir,
        new_dir=None,
        name="Run",
        test_id="lpips",
        in_chans=2,
        crop_size=256,
        stride=128,
        images_to_use="both",
        device=torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
):
    if new_dir is None:
        parent_dir = os.path.dirname(dir)
        new_dir = os.path.join(parent_dir, f"{name}_predicted_images")
        os.makedirs(new_dir, exist_ok=True)
    G = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=in_chans,           # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
    )
    G.load_state_dict(torch.load(f"best_model_G_stain_{test_id}_{name}.pth"))
    G.eval()


    img_list = []
    img_path_list = os.listdir(dir)
    for j in range(len(img_path_list)):
        image = tifffile.imread(os.path.join(dir, img_path_list[j]))
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

    for j in tqdm(range(len(img_path_list))):
        image = tifffile.imread(os.path.join(dir,img_path_list[j]))
        for l in range(len(image[0][0])):
            image[..., l] = stand_norm(image[..., l], img_minmax[l])
        np.concatenate([image, np.zeros((*image.shape[:2], 1), dtype=image.dtype)], axis=2)
        with torch.no_grad():
            image_both = np.transpose(torch.tensor(image[..., 0:2]), (2, 0, 1)).unsqueeze(0)
            pred = F.sigmoid(G(image_both))
            pred = pred[0, 0].numpy()
            image[..., 2] = pred
            tifffile.imwrite(os.path.join(new_dir, img_path_list[j]), image)
    print("predict_images done")