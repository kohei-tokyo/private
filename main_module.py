import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from pytorch_msssim import ssim
from dataclasses import asdict
import timm
import torch.nn.functional as F
import lpips
import itertools
import wandb
import copy
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel,
    DDPMScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer
# from tqdm.auto import tqdm
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import os
import torchvision.transforms as T
from diffusers import StableDiffusionControlNetPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from make_dataset import DatasetDigitalStaining
from utils import *
from config import *

class ControlNet(nn.Module):
    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        # --- パスとデータに関する設定 ---
        self.dir = self.config.dir
        self.train_folders = self.config.train_folders
        self.val_folders = self.config.val_folders
        self.test_folders = self.config.test_folders
        self.target = self.config.target
        self.images_to_use = self.config.images_to_use

        # --- 実験管理 (W&Bなど) ---
        self.name = self.config.name
        self.group = self.config.group

        # --- 学習ループに関する設定 ---
        self.n_epoch = self.config.n_epoch
        self.learning_rate = self.config.learning_rate
        self.batch_size = self.config.batch_size
        self.patches_per_epoch = self.config.patches_per_epoch
        self.patches_per_epoch_val = self.config.patches_per_epoch_val
        self.val_epoch = self.config.val_epoch
        self.gradient_accumulation_steps = self.config.gradient_accumulation_steps
        self.learning_rate_scheduler = self.config.learning_rate_scheduler
        # self.w_ssim = self.config.w_ssim

        # # --- UNetモデルのアーキテクチャ設定 ---
        # self.model_unet = self.config.model_unet
        # self.in_chans = self.config.in_chans
        # self.dim_mults = self.config.dim_mults
        # self.real_pred = self.config.real_pred

        # --- 拡散過程に関する設定 ---
        self.noise_steps = self.config.noise_steps
        # self.noise_add = self.config.noise_add

        # --- 推論に関する設定 ---
        self.img_size = self.config.img_size
        # self.pred_size = self.config.pred_size
        self.num_inference_steps = self.config.num_inference_steps
        # self.mode_dif = self.config.mode_dif
        # self.cfg_scale = self.config.cfg_scale
        # self.no_label = self.config.no_label
        self.val_crop = self.config.val_crop

        # --- 環境設定 ---
        self.num_workers = self.config.num_workers
        self.device = self.config.device


        self.target_class = len(self.target)

        # --- 1. デバイスと言語モデル設定 ---
        self.model_id = "runwayml/stable-diffusion-v1-5"
        self.model_dtype = torch.float16  # VRAM節約のため混合精度

        # --- 2. ノイズスケジューラの読み込み ---
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.model_id, subfolder="scheduler")

        # --- 3. トークナイザとテキストエンコーダ ---
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_id, subfolder="text_encoder", dtype=self.model_dtype
        ).to(self.device)
        self.text_encoder.eval()

        # --- 4. VAE (Autoencoder) ---
        self.vae = AutoencoderKL.from_pretrained(
            self.model_id, subfolder="vae", torch_dtype=self.model_dtype
        ).to(self.device)
        self.vae.eval()

        # --- 5. ベースのU-Net (凍結対象) ---
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet", torch_dtype=self.model_dtype
        ).to(self.device)
        self.unet.eval()

        # --- 6. ControlNet (学習対象) ---
        # SDのU-Netの重みをコピーして初期化
        self.controlnet = ControlNetModel.from_unet(self.unet).to(self.device)

        # --- 7. ここが最重要：ControlNet以外をすべて凍結 ---
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        # controlnet.requires_grad_(True) はデフォルトなのでOK

        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.train_loader = self.make_loader(self.train_folders, "train")
        self.val_loader = self.make_loader(self.val_folders, "val")
        self.test_loader = self.make_loader(self.test_folders, "test")

        # self.controlnet.to(self.model_dtype)
        self.optimizer = AdamW(
            self.controlnet.parameters(),
            lr=self.learning_rate
        )
        if self.learning_rate_scheduler:
            num_training_steps = self.patches_per_epoch * self.n_epoch
            self.lr_scheduler = get_scheduler(
                "constant",
                optimizer=self.optimizer,
                num_warmup_steps=int(num_training_steps * 0.05),
                num_training_steps=(num_training_steps)
            )

        self.scaler = torch.amp.GradScaler(device=self.device)

        self.pipeline = StableDiffusionControlNetPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.noise_scheduler,
            controlnet=self.controlnet,
            safety_checker=None,
            feature_extractor=None,
        )

        # 2. (Line 164) EMAの「パラメータ保管庫」を作成
        self.ema_controlnet = EMAModel(
            self.controlnet.parameters(),  # .parameters() を渡す
            decay=0.9999
        )

        # 3. EMAの重みを受け取るための「専用モデル」をディープコピーで作成
        self.ema_controlnet_model = copy.deepcopy(self.controlnet).to(self.device)

        # 4. EMA専用のパイプラインを初期化
        #    controlnet 引数に「EMA専用モデル」を渡す
        self.ema_pipeline = StableDiffusionControlNetPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.noise_scheduler,
            controlnet=self.ema_controlnet_model,  # <-- EMA専用モデルを指定
            safety_checker=None,
            feature_extractor=None,
        )
        self.ema_pipeline = self.ema_pipeline.to(self.device, dtype=self.model_dtype)


        # self.ema_controlnet = EMAModel(self.controlnet.parameters(), decay=0.9999)
        # # 1. EMAモデル本体を評価 (eval) モードに設定
        # # self.ema_controlnet.model.eval()
        #
        # # 2. EMAモデル（ema_controlnet.model）を使って、
        # #    新しいパイプラインを「手動でインスタンス化」します
        # self.ema_pipeline = StableDiffusionControlNetPipeline(
        #     vae=self.vae,
        #     text_encoder=self.text_encoder,
        #     tokenizer=self.tokenizer,
        #     unet=self.unet,
        #     scheduler=self.noise_scheduler,
        #     controlnet=self.ema_controlnet,  # <-- EMAモデルをここで指定
        #     safety_checker=None,
        #     feature_extractor=None,
        # )
        #
        # # 3. パイプライン全体をGPUに送り、型を設定
        # self.ema_pipeline = self.ema_pipeline.to(self.device, dtype=self.model_dtype)

        self.loss_fn_mse = nn.MSELoss().to(self.device)
        self.loss_fn_ssim = tensor_ssim
        self.loss_fn_lpips = lpips.LPIPS(net='alex').to(self.device)
        self.hist = {"train": [], "val": [], "test": []}
        self.val_list = [
            "ssim",
            "lpips",
            "ssim_ema",
            "lpips_ema",
        ]
        self.best_score_list = [float('inf')] * len(self.val_list)
        self.min_epoch_list = [0] * len(self.val_list)
        self.epoch = 0

        self.Result = Result(
            self.model_sample,
            self.model_sample_ema,
            self.train_loader,
            self.val_loader,
            self.test_loader,
            self.img_size,
            target=self.target,
            val_crop=self.val_crop,
            min_max=[-1, 1],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )


    def make_loader(self, folders, mode):
        datasets = []
        for target_n in range(self.target_class):
            for f in folders[target_n]:
                img_folder = os.path.join(self.dir[target_n], f)
                datasets.append(
                    DatasetDigitalStaining(img_folder, self.target[target_n], tokenizer=self.tokenizer, RGB_channels=True,
                                           augmentation=None))
        combined_dataset = ConcatDataset(datasets)
        if mode == "train":
            return DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=True,
                              num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0)
        else:
            return DataLoader(combined_dataset, batch_size=4, shuffle=False,
                              num_workers=0, pin_memory=True)


    def _wandb_init(self):
        self.run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="kohei_tokyo-the-university-of-tokyo",
            # Set the wandb project where this run will be logged.
            project="Diffusion_model",
            name=self.name,
            group=self.group,
            # Track hyperparameters and run metadata.
            config=asdict(self.config),
        )


    def all(self):
        self.train()
        self.test()


    def train(self):
        self._wandb_init()
        for self.epoch in range(self.n_epoch):
            print(f"Epoch {self.epoch + 1}/{self.n_epoch}")
            self.calc_epoch("train")
            if self.epoch % self.val_epoch == 0:
                self.calc_epoch("val")
                self.Result.show_result("val", epoch=self.epoch)

    def test(self):
        test_list = ["final", "ssim", "lpips"]
        # test_list = ["lpips"]
        for self.test_id in test_list:
            print(f"Test {self.test_id}")
            self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                self.model_id,
                controlnet=ControlNetModel.from_pretrained(f"path//best_model_stain_{self.test_id}_{self.name}.pth").to(self.model_dtype),
                torch_dtype=self.model_dtype,
            ).to(self.device)
            self.ema_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                self.model_id,
                controlnet=ControlNetModel.from_pretrained(f"path//best_model_stain_{self.test_id}_ema_{self.name}.pth").to(
                    self.model_dtype),
                torch_dtype=self.model_dtype,
            ).to(self.device)
            self.calc_epoch("test")
            self.Result.show_result("test", test_id=self.test_id)
        wandb.finish()

    def calc_epoch(self, mode):
        if mode == "train":
            self.controlnet.train()
            # self.ema_controlnet.model.train()
            loader = itertools.islice(self.train_loader, self.patches_per_epoch)
            patches_num = self.patches_per_epoch
        elif mode == "val":
            self.controlnet.eval()
            # self.ema_controlnet.model.eval()
            # loader = itertools.islice(self.val_loader, self.patches_per_epoch_val)
            # patches_num = self.patches_per_epoch_val
            loader = self.val_loader
            patches_num = len(loader)
        elif mode == "test":
            self.controlnet.eval()
            # self.ema_controlnet.model.eval()
            # loader = itertools.islice(self.test_loader, self.patches_per_epoch_val)
            # patches_num = self.patches_per_epoch_val
            loader = self.test_loader
            patches_num = len(loader)
        else:
            raise NotImplementedError

        total_metrics_dict = {}
        n_target_sum = {}
        # progress_bar = tqdm(total=patches_num, desc=mode)
        for step, batch in tqdm(enumerate(loader), total=patches_num, desc=mode):
        # for step, batch in enumerate(loader):
            metrics, n_target = self.calc_batch(step, batch, mode)
            # if total_metrics_dict is None:
            #     total_metrics_dict = {k: 0 for k, v in metrics.item()}
            # for k, v in metrics.items():
            #     total_metrics_dict[k] += metrics[k].items()

            # if total_metrics_dict is None:
            #     total_metrics_dict = metrics.copy()
            # if n_target_sum is None:
            #     if n_target is not None:
            #         n_target_sum = n_target.copy()
            # else:
            #     for k, v in metrics.items():
            #         total_metrics_dict[k] = total_metrics_dict.get(k, 0) + v
            #     if n_target is not None:
            #         for k, v in n_target.items():
            #             n_target_sum[k] = n_target_sum.get(k, 0) + v

            # 辞書に加算する際、型を Python float に統一する
            for k, v in metrics.items():
                # v が Tensor (テンソル) か、
                # それ以外 (float) かを判定
                if isinstance(v, torch.Tensor):
                    # v がテンソルの場合、.item() を呼んで float に変換
                    value_to_add = v.item()
                else:
                    # v が既に float の場合、そのまま使用
                    value_to_add = v
                # 常に float 型の value_to_add を加算
                total_metrics_dict[k] = total_metrics_dict.get(k, 0) + value_to_add

            # n_target は常に int (count) なので、そのままでOK
            if n_target is not None:
                for k, v in n_target.items():
                    n_target_sum[k] = n_target_sum.get(k, 0) + v

        for k, v in total_metrics_dict.items():
            if n_target is not None:
                if k in n_target_sum:
                    count = n_target_sum[k]
                    total_metrics_dict[k] = v / (count + 1e-5)
                else:
                    total_metrics_dict[k] = v / patches_num
            else:
                total_metrics_dict[k] = v / patches_num

        self.save_results(total_metrics_dict, mode)

    def calc_batch(self, step, batch, mode):
        if mode == "train":
            return self.calc_matrix_train(step, batch)
        else:
            target_images = batch["target_image"].to(self.device)
            conditioning_images = batch["conditioning_image"].to(self.device)
            prompt = batch["prompt"]

            metrics_dict = None
            len_met = 0
            _, _, H, W = target_images.shape
            n = self.patches_per_epoch_val
            for y_i in range(H // 2, H - self.img_size + 1, self.img_size):
                for x_i in range(W // 2, W - self.img_size + 1, self.img_size):
                    len_met += 1
                    # print(ph1[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size].shape)
                    metrics, n_target = self.calc_matrix_test(
                        target_images[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size],
                        conditioning_images[:, :, y_i:y_i + self.img_size, x_i:x_i + self.img_size],
                        prompt
                    )
                    # if metrics_dict is None:
                    #     metrics_dict = {k: 0 for k, v in metrics.items()}
                    #     n_target_sum = n_target
                    # for k, v in metrics.items():
                    #     metrics_dict[k] += metrics[k]
                    # if n_target_sum is None:
                    #     n_target_sum = n_target.copy()
                    # else:
                    #     for k, v in n_target.items():
                    #         n_target_sum[k] = n_target_sum.get(k, 0) + v
                    if metrics_dict is None:
                        metrics_dict = metrics.copy()
                        n_target_sum = n_target.copy()
                    else:
                        for k, v in metrics.items():
                            metrics_dict[k] = metrics_dict.get(k, 0) + v
                        for k, v in n_target.items():
                            n_target_sum[k] = n_target_sum.get(k, 0) + v
                    n = n - 1
                    if n <= 0:
                        break
                if n <= 0:
                    break

            for k, v in metrics_dict.items():
                # metrics_done = False
                # for c in self.target:
                #     if k.split("_")[-1] == c:
                #         metrics_dict[k] /= n_target_sum[c]
                #         metrics_done = True
                # if not metrics_done:
                metrics_dict[k] /= len_met
            return metrics_dict, n_target_sum

    def calc_matrix_train(self, step, batch):
        # 混合精度 (FP16) のコンテキスト
        with torch.amp.autocast(device_type=self.device.type, dtype=self.model_dtype):
            # --- 1. バッチデータをデバイスへ ---
            target_images = batch["target_image"].to(self.device)
            conditioning_images = batch["conditioning_image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)

            # --- 2. VAEエンコード ---
            # target_images (RGB) -> latents (潜在変数)
            # .sample() で分布からサンプリング
            latents = self.vae.encode(target_images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor  # スケーリング

            # --- 3. Textエンコード ---
            encoder_hidden_states = self.text_encoder(input_ids)[0]

            # --- 4. ノイズの追加 (Forward process) ---
            # バッチ内の各画像にランダムなノイズ
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # ランダムなタイムステップ (t) を選択
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device)
            timesteps = timesteps.long()

            # latentsにノイズを追加 -> noisy_latents
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            # --- 5. ControlNet + U-Net によるノイズ予測 (Denoising process) ---

            # 5a. ControlNetが「条件画像」からヒントを抽出
            controlnet_output = self.controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=conditioning_images,  # ここに光学画像
                return_dict=True,
            )
            down_block_res_samples = controlnet_output.down_block_res_samples
            mid_block_res_sample = controlnet_output.mid_block_res_sample

            # 5b. U-Netが「ノイズ」と「ControlNetのヒント」を使ってノイズを予測
            model_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,  # ControlNetの出力を渡す
                mid_block_additional_residual=mid_block_res_sample,  # ControlNetの出力を渡す
            ).sample

            # --- 6. 損失の計算 ---
            # 予測したノイズ (model_pred) と、実際に追加したノイズ (noise) の差
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            # (VRAMが厳しい場合) 勾配を蓄積
            loss = loss / self.gradient_accumulation_steps

        # --- 7. バックプロパゲーション ---
        # scaler.scale(loss) -> 混合精度用のスケーリング
        self.scaler.scale(loss).backward()

        # 蓄積ステップ数に達したらパラメータを更新
        if (step + 1) % self.gradient_accumulation_steps == 0:
            # 勾配のクリッピング (発散防止)
            torch.nn.utils.clip_grad_norm_(self.controlnet.parameters(), 1.0)

            self.scaler.step(self.optimizer)  # オプティマイザ実行
            self.scaler.update()  # スケーラー更新
            if self.learning_rate_scheduler:
                self.lr_scheduler.step()  # (スケジューラを使う場合)
            # controlnet の最新の重みを ema_controlnet に反映
            # self.ema_controlnet.step(self.controlnet.parameters())
            self.ema_controlnet.step(self.controlnet.parameters())
            self.optimizer.zero_grad()  # 勾配をリセット

        output_dict = {"mse": loss}
        return output_dict, None

    def calc_matrix_test(self, target_images, conditioning_images, prompt):
        # prediction = self.model_sample(conditioning_images, prompt).to(dtype=target_images.dtype)
        # mse_loss = self.loss_fn_mse(target_images, prediction)
        # ssim_loss = self.loss_fn_ssim(target_images, prediction)
        # lpips_loss = self.loss_fn_lpips(target_images, prediction).mean()
        #
        # ema_prediction = self.model_sample_ema(conditioning_images, prompt).to(dtype=target_images.dtype)
        # mse_loss_ema = self.loss_fn_mse(target_images, ema_prediction)
        # ssim_loss_ema = self.loss_fn_ssim(target_images, ema_prediction)
        # lpips_loss_ema = self.loss_fn_lpips(target_images, ema_prediction).mean()

        prediction = self.model_sample(conditioning_images, prompt).to(dtype=target_images.dtype)
        # l1_loss = self.l1_loss(target_images, prediction)
        ssim_loss = self.loss_fn_ssim(target_images, prediction)
        # mse = self.loss_fn_mse(target_images, prediction)
        lpips_loss, n_target = lpips_calc_each(target_images, prediction, prompt, self.target, self.loss_fn_lpips, "lpips")

        ema_prediction = self.model_sample_ema(conditioning_images, prompt).to(dtype=target_images.dtype)
        # l1_loss_ema = self.l1_loss(target_images, ema_prediction)
        ssim_loss_ema = self.loss_fn_ssim(target_images, ema_prediction)
        # mse_ema = self.loss_fn_mse(target_images, ema_prediction)
        lpips_loss_ema, n_target_ema = lpips_calc_each(target_images, ema_prediction, prompt, self.target, self.loss_fn_lpips, "lpips_ema")

        loss_dict = {"ssim": ssim_loss, "ssim_ema": ssim_loss_ema}
        # loss_dict = {
        #     "l1_loss": l1_loss, "mse": mse, "ssim": ssim_loss,
        #     "l1_loss_ema": l1_loss_ema, "ssim_ema": ssim_loss_ema, "mse_ema": mse_ema,
        # }
        loss_dict = dict(**loss_dict, **lpips_loss, **lpips_loss_ema)

        return loss_dict, n_target

    def model_sample(self, conditioning_images, prompt):
        self.controlnet.eval()
        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type, dtype=self.model_dtype):
            # 1. pipelineは "pt" を指定しても CPU テンソルを返す
            images = self.pipeline(
                prompt,
                image=conditioning_images,
                num_inference_steps=self.num_inference_steps,
                output_type="latent",
                disable_tqdm=True
            ).images
        with torch.no_grad():
            images = images / self.vae.config.scaling_factor
            images = self.vae.decode(images.to(self.vae.dtype)).sample
            images = (images / 2 + 0.5).clamp(0, 1)  # [0, 1]に正規化
        return images

    def model_sample_ema(self, conditioning_images, prompt):
        # --- ★★★ 修正箇所 ★★★ ---
        # 1. EMA「保管庫」(storage) から「EMAモデル」に最新の重みをコピー
        self.ema_controlnet.copy_to(self.ema_controlnet_model.parameters())

        # 2. EMAモデルを評価モードに (念のため)
        self.ema_controlnet_model.eval()

        # 3. EMA専用パイプラインで推論を実行 (autocast を使用)
        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type, dtype=self.model_dtype):
            latents_gpu = self.ema_pipeline(
                prompt,
                image=conditioning_images,
                num_inference_steps=self.num_inference_steps,
                output_type="latent",
                disable_tqdm=True
            ).images  # (前回の修正 .latents -> .images を適用済み)

        # 4. VAE デコード (autocast の外)
        with torch.no_grad():
            latents_gpu = latents_gpu / self.vae.config.scaling_factor
            images_gpu = self.vae.decode(latents_gpu.to(self.vae.dtype)).sample
            images_gpu = (images_gpu / 2 + 0.5).clamp(0, 1)  # [0, 1]に正規化

        return images_gpu


    def save_results(self, total_metrics_dict, mode):
        self.hist[mode].append(total_metrics_dict)
        for k, v in total_metrics_dict.items():
            print(f"{mode} {k}: {v}")
        if mode == "test":
            if self.test_id == "final":
                total_metrics_dict_data = [[k, v] for k, v in total_metrics_dict.items()]
                wandb.log(
                    {f"{mode}/{self.test_id}": wandb.Table(data=total_metrics_dict_data, columns=["index", "score"])})
            else:
                self.run.log({f"{mode}/{self.test_id}": total_metrics_dict[self.test_id],
                              "ema": 0})
                self.run.log({f"{mode}/{self.test_id}": total_metrics_dict[f"{self.test_id}_ema"],
                              "ema": 1})
            if self.test_id == "lpips":
                self.run.summary["lpips"] = total_metrics_dict["lpips"]
                self.run.summary["lpips_ema"] = total_metrics_dict["lpips_ema"]
                for c in self.target:
                    self.run.log({f"{mode}/lpips_{c}": total_metrics_dict[f"lpips_{c}"],
                                  "ema": 0})
                    self.run.log({f"{mode}/lpips_{c}": total_metrics_dict[f"lpips_ema_{c}"],
                                  "ema": 1})
        else:
            total_metrics_dict_log = {f"{mode}/" + k: v for k, v in total_metrics_dict.items()}
            epoch_dict = {"epoch": self.epoch}
            self.run.log(dict(**total_metrics_dict_log, **epoch_dict))

        if mode == "val":
            self.controlnet.save_pretrained(f"path//best_model_stain_final_{self.name}.pth")
            self.ema_controlnet_model.save_pretrained(f"path//best_model_stain_final_ema_{self.name}.pth")
            for val_n in range(len(self.val_list)):
                mean_loss = total_metrics_dict[self.val_list[val_n]]
                if mean_loss < self.best_score_list[val_n]:
                    print(f"Loss_{self.val_list[val_n]} improved to {mean_loss}, saving model")
                    self.best_score_list[val_n] = mean_loss
                    self.min_epoch_list[val_n] = self.epoch
                    if val_n < len(self.val_list) / 2:
                        self.controlnet.save_pretrained(f"path//best_model_stain_{self.val_list[val_n]}_{self.name}.pth")
                    else:
                        self.ema_controlnet_model.save_pretrained(f"path//best_model_stain_{self.val_list[val_n]}_{self.name}.pth")