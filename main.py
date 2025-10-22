from GAN import StainingGAN
from predict_images import PredictGAN
import os
import torch
from pathlib import Path

class DigitalStaining():
    def __init__(
            self,
            dir=None,  # train or test : 元データのpath, predict : 入力画像のpath
            main_dir=None,  # train or test : 前処理データ保存フォルダのpath, predict : なし
            original_dir=None,  # train or test : なし, predict : 入力画像のpath
            new_dir=None,  # train or test : なし, predict : 予測画像保存フォルダのpath
            name="Run",  # 名称
            produce_image=False,  # 一度前処理をしている場合はFalse
            train_folders=["train"],  # Trainデータのフォルダ名
            val_folders=["val"],  # Valデータのフォルダ名
            test_folders=["test"],  # Testデータのフォルダ名
            target=None, # "original_ER", "preprocess_ER", "original_mito", "preprocess_mito"
            # produce_images
            img_n=100,
            img_size=256,
            # gan
            n_epoch=10,
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
            # predict
            test_id="lpips"
    ):
        self.dir = Path(dir)
        self.main_dir = main_dir
        self.original_dir = original_dir
        self.new_dir = new_dir
        self.name = name
        self.produce_image = produce_image
        self.train_folders = train_folders
        self.val_folders = val_folders
        self.test_folders = test_folders
        self.img_n = img_n
        self.img_size = img_size
        self.n_epoch = n_epoch
        self.discriminator = discriminator
        self.num_workers = num_workers
        self.in_chans = in_chans
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_dice = w_dice
        self.img_size = img_size
        self.crop_size = crop_size
        self.stride = stride
        self.learning_rate_g = learning_rate_g
        self.learning_rate_d = learning_rate_d
        self.betas = betas
        self.images_to_use = images_to_use
        self.device = device
        self.patches_per_epoch = patches_per_epoch
        self.val_epoch = val_epoch
        self.batch_size = batch_size
        self.test_id = test_id

        # if target is not None:
        if target == "preprocess_mito":
            self.train_folders = ["2", "3", "4"]
            self.val_folders = ["5"]
            self.test_folders = ["1"]
            self.dir = "D:\\Matsusaka\\data_mito\\HeLa_Su9-mSG_UNetPipeline_crop"
        elif target == "original_mito":
            self.train_folders = ["2", "3", "4"]
            self.val_folders = ["5"]
            self.test_folders = ["1"]
            self.dir = "D:\\Matsusaka\\data_mito\\HeLa_Su9-mSG_original_crop"
        elif target == "preprocess_ER":
            self.train_folders = ["1"]
            self.val_folders = ["2"]
            self.test_folders = ["3"]
            self.dir = "D:\\Matsusaka\\data_mito\\COS7_KDEL-mSG_UNetPipeline_crop"
        elif target == "original_ER":
            self.train_folders = ["1"]
            self.val_folders = ["2"]
            self.test_folders = ["3"]
            self.dir = "D:\\Matsusaka\\data_mito\\COS7_KDEL-mSG_original_crop"

    def _init_gan(self):
        self.gan = StainingGAN(
            self.main_dir,
            train_folders=self.train_folders,
            val_folders=self.val_folders,
            test_folders=self.test_folders,
            name=self.name,
            n_epoch=self.n_epoch,
            discriminator=self.discriminator,  # Patch4, Patch3, Patch5, ResnetPatch, Resnet, or U_Net
            num_workers=self.num_workers,  # GPUのメモリが足りない場合は小さくしてください
            in_chans=self.in_chans,
            w_l1=self.w_l1,
            w_ssim=self.w_ssim,
            w_dice=self.w_dice,
            crop_size=self.crop_size,
            stride=self.stride,
            learning_rate_g=self.learning_rate_g,
            learning_rate_d=self.learning_rate_d,
            betas=self.betas,
            images_to_use=self.images_to_use,
            device=self.device,
            patches_per_epoch=self.patches_per_epoch,
            val_epoch=self.val_epoch,
            batch_size=self.batch_size
        )


    def all(self):
        if self.produce_image:
            if self.main_dir is None:
                parent_dir = os.path.dirname(self.dir)
                self.main_dir = os.path.join(parent_dir, f"{self.name}_produced_images")
                os.makedirs(self.main_dir, exist_ok=True)
            from produce_images import produce_images
            produce_images(
                self.dir,
                self.main_dir,
                train_folders=self.train_folders,
                val_folders=self.val_folders,
                test_folders=self.test_folders,
                img_n=self.img_n,
                img_size=self.img_size
            )
        else:
            self.main_dir = self.dir
        self._init_gan()
        self._train()
        self._test()

    def train(self):
        if self.produce_image:
            if self.main_dir is None:
                parent_dir = os.path.dirname(self.dir)
                self.main_dir = os.path.join(parent_dir, f"{self.name}_produced_images")
                os.makedirs(self.main_dir, exist_ok=True)
            from produce_images import produce_images
            produce_images(
                self.dir,
                self.main_dir,
                train_folders=self.train_folders,
                val_folders=self.val_folders,
                test_folders=self.test_folders,
                img_n=self.img_n,
                img_size=self.img_size
            )
        else:
            self.main_dir = self.dir

        self._init_gan()
        self._train()

    def test(self):
        if self.produce_image:
            if self.main_dir is None:
                parent_dir = os.path.dirname(self.dir)
                self.main_dir = os.path.join(parent_dir, f"{self.name}_produced_images")
                os.makedirs(self.main_dir, exist_ok=True)
        else:
            self.main_dir = self.dir
        self._init_gan()

    def _train(self):
        self.gan.train()

    def _test(self):
        self.gan.test()

    def predict(self):
        if self.original_dir is None:
            self.original_dir = self.dir
        PredictGAN(
            self.original_dir,
            new_dir=self.new_dir,
            name=self.name,
            test_id=self.test_id,
            in_chans=self.in_chans,
            crop_size=self.crop_size,
            stride=self.stride,
            images_to_use=self.images_to_use,
            device=self.device
        )


import argparse

if __name__ == '__main__':
    # --- 全体の説明 ---
    parser = argparse.ArgumentParser(
        description='Digital Staining',
        formatter_class=argparse.RawTextHelpFormatter
    )
    # 実行モード (train/test/predict) を選択
    subparsers = parser.add_subparsers(dest='mode', required=True, help='実行モードを選択してください')

    # --- 学習 (train) モードの引数 ---
    parser_train = subparsers.add_parser('train', help='モデルの学習を開始します')
    parser_train.add_argument('--dir', type=str, required=True, help='元データ（train/val/testフォルダを含む）のパス')
    parser_train.add_argument('--name', type=str, default='Run', help='学習の名称 (モデルの保存名などに使われます)')
    parser_train.add_argument('--n_epoch', type=int, default=50, help='学習のエポック数')
    parser_train.add_argument('--batch_size', type=int, default=16, help='バッチサイズ')
    parser_train.add_argument('--num_workers', type=int, default=4, help='データ読み込みの並列プロセス数 (メモリ不足の場合は小さくしてください)')
    parser_train.add_argument('--no_produce_image', action='store_false', dest='produce_image', help='このフラグを立てると画像の前処理・拡張を行いません')
    parser_train.add_argument('--main_dir', type=str, default=None, help='前処理済み画像の保存先 (指定しない場合は自動生成されます)')

    # --- テスト (test) モードの引数 ---
    parser_test = subparsers.add_parser('test', help='学習済みモデルの評価を行います')
    parser_test.add_argument('--dir', type=str, required=True, help='評価用データセットのパス')
    parser_test.add_argument('--name', type=str, default='Run', help='評価したい学習の名称')
    # testモードではデフォルトで画像前処理をオフにする
    parser_test.set_defaults(produce_image=False)


    # --- 予測 (predict) モードの引数 ---
    parser_predict = subparsers.add_parser('predict', help='新しい画像に対して予測を行います')
    parser_predict.add_argument('--original_dir', type=str, required=True, help='予測したい入力画像のディレクトリパス')
    parser_predict.add_argument('--new_dir', type=str, required=True, help='予測画像の保存先ディレクトリパス')
    parser_predict.add_argument('--name', type=str, default='Run', help='使用する学習済みモデルの名称')
    parser_predict.add_argument('--test_id', type=str, default='lpips', choices=['mse', 'ssim', 'lpips'],
                                help='使用するモデルの重みを選択 (どの評価指標で最適化されたモデルか)')

    # --- 引数の解析と実行 ---
    args = parser.parse_args()

    # DigitalStainingクラスのインスタンス化
    # DigitalStainingクラスの__init__に存在しない引数をargsから削除
    digital_staining_args = vars(args).copy()
    allowed_args = DigitalStaining.__init__.__code__.co_varnames
    for key in list(digital_staining_args.keys()):
        if key not in allowed_args:
            del digital_staining_args[key]

    stain = DigitalStaining(**digital_staining_args)

    # 選択されたモードに応じて実行
    if args.mode == 'train':
        stain.train()
    elif args.mode == 'test':
        stain.test()
    elif args.mode == 'predict':
        # predictモードではdir引数が不要なため、クラス初期化後に設定
        stain.original_dir = args.original_dir
        stain.new_dir = args.new_dir
        stain.predict()