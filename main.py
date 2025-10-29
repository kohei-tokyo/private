from GAN import StainingGAN
from config import DigitalStainingConfig
from predict_images import PredictGAN
import os
import torch
from pathlib import Path

class DigitalStaining:
    def __init__(self, config: DigitalStainingConfig):
        self.config = config

        # if target is not None:
        if self.config.target == "mito":
            self.config.train_folders = ["2", "3", "4"]
            self.config.val_folders = ["5"]
            self.config.test_folders = ["1"]
            if self.config.img_size == 256:
                self.config.dir = r"C:\data\mito_crop_256"
            elif self.config.img_size == 128:
                self.config.dir = r"C:\data\mito_crop_128"
        elif self.config.target == "ER":
            self.config.train_folders = ["1"]
            self.config.val_folders = ["2"]
            self.config.test_folders = ["3"]
            if self.config.img_size == 256:
                self.config.dir = r"C:\data\ER_crop_256"
            elif self.config.img_size == 128:
                self.config.dir = r"C:\data\ER_crop_128"

    def _init_gan(self):
        self.gan = StainingGAN(config = self.config)

    def all(self):
        # if self.config.produce_image:
        #     if self.config.main_dir is None:
        #         parent_dir = os.path.dirname(self.config.dir)
        #         self.config.main_dir = os.path.join(parent_dir, f"{self.config.name}_produced_images")
        #         os.makedirs(self.config.main_dir, exist_ok=True)
        #     from produce_images import produce_images
        #     produce_images(
        #         self.config.dir,
        #         self.config.main_dir,
        #         train_folders=self.config.train_folders,
        #         val_folders=self.config.val_folders,
        #         test_folders=self.config.test_folders,
        #         img_n=self.config.img_n,
        #         image_size=self.config.image_size
        #     )
        # else:
        #     self.config.main_dir = self.config.dir
        self._init_gan()
        self._train()
        self._test()

    def train(self):
        # if self.config.produce_image:
        #     if self.config.main_dir is None:
        #         parent_dir = os.path.dirname(self.config.dir)
        #         self.config.main_dir = os.path.join(parent_dir, f"{self.config.name}_produced_images")
        #         os.makedirs(self.main_dir, exist_ok=True)
        #     from produce_images import produce_images
        #     produce_images(
        #         self.config.dir,
        #         self.config.main_dir,
        #         train_folders=self.config.train_folders,
        #         val_folders=self.config.val_folders,
        #         test_folders=self.config.test_folders,
        #         img_n=self.config.img_n,
        #         img_size=self.config.img_size
        #     )
        # else:
        #     self.config.main_dir = self.config.dir

        self._init_gan()
        self._train()

    def test(self):
        # if self.config.produce_image:
        #     if self.config.main_dir is None:
        #         parent_dir = os.path.dirname(self.config.dir)
        #         self.main_dir = os.path.join(parent_dir, f"{self.config.name}_produced_images")
        #         os.makedirs(self.config.main_dir, exist_ok=True)
        # else:
        #     self.config.main_dir = self.config.dir
        self._init_gan()
        self._test()

    def _train(self):
        self.gan.train()

    def _test(self):
        self.gan.test()

    def predict(self):
        if self.config.original_dir is None:
            self.config.original_dir = self.config.dir
        PredictGAN(
            self.config.original_dir,
            new_dir=self.config.new_dir,
            name=self.config.name,
            test_id=self.config.test_id,
            in_chans=self.config.in_chans,
            crop_size=self.config.crop_size,
            stride=self.config.stride,
            images_to_use=self.config.images_to_use,
            device=self.config.device
        )


# import argparse
#
# if __name__ == '__main__':
#     # --- 全体の説明 ---
#     parser = argparse.ArgumentParser(
#         description='Digital Staining',
#         formatter_class=argparse.RawTextHelpFormatter
#     )
#     # 実行モード (train/test/predict) を選択
#     subparsers = parser.add_subparsers(dest='mode', required=True, help='実行モードを選択してください')
#
#     # --- 学習 (train) モードの引数 ---
#     parser_train = subparsers.add_parser('train', help='モデルの学習を開始します')
#     parser_train.add_argument('--dir', type=str, required=True, help='元データ（train/val/testフォルダを含む）のパス')
#     parser_train.add_argument('--name', type=str, default='Run', help='学習の名称 (モデルの保存名などに使われます)')
#     parser_train.add_argument('--n_epoch', type=int, default=50, help='学習のエポック数')
#     parser_train.add_argument('--batch_size', type=int, default=16, help='バッチサイズ')
#     parser_train.add_argument('--num_workers', type=int, default=4, help='データ読み込みの並列プロセス数 (メモリ不足の場合は小さくしてください)')
#     parser_train.add_argument('--no_produce_image', action='store_false', dest='produce_image', help='このフラグを立てると画像の前処理・拡張を行いません')
#     parser_train.add_argument('--main_dir', type=str, default=None, help='前処理済み画像の保存先 (指定しない場合は自動生成されます)')
#
#     # --- テスト (test) モードの引数 ---
#     parser_test = subparsers.add_parser('test', help='学習済みモデルの評価を行います')
#     parser_test.add_argument('--dir', type=str, required=True, help='評価用データセットのパス')
#     parser_test.add_argument('--name', type=str, default='Run', help='評価したい学習の名称')
#     # testモードではデフォルトで画像前処理をオフにする
#     parser_test.set_defaults(produce_image=False)
#
#
#     # --- 予測 (predict) モードの引数 ---
#     parser_predict = subparsers.add_parser('predict', help='新しい画像に対して予測を行います')
#     parser_predict.add_argument('--original_dir', type=str, required=True, help='予測したい入力画像のディレクトリパス')
#     parser_predict.add_argument('--new_dir', type=str, required=True, help='予測画像の保存先ディレクトリパス')
#     parser_predict.add_argument('--name', type=str, default='Run', help='使用する学習済みモデルの名称')
#     parser_predict.add_argument('--test_id', type=str, default='lpips', choices=['mse', 'ssim', 'lpips'],
#                                 help='使用するモデルの重みを選択 (どの評価指標で最適化されたモデルか)')
#
#     # --- 引数の解析と実行 ---
#     args = parser.parse_args()
#
#     # DigitalStainingクラスのインスタンス化
#     # DigitalStainingクラスの__init__に存在しない引数をargsから削除
#     digital_staining_args = vars(args).copy()
#     allowed_args = DigitalStaining.__init__.__code__.co_varnames
#     for key in list(digital_staining_args.keys()):
#         if key not in allowed_args:
#             del digital_staining_args[key]
#
#     stain = DigitalStaining(**digital_staining_args)
#
#     # 選択されたモードに応じて実行
#     if args.mode == 'train':
#         stain.train()
#     elif args.mode == 'test':
#         stain.test()
#     elif args.mode == 'predict':
#         # predictモードではdir引数が不要なため、クラス初期化後に設定
#         stain.original_dir = args.original_dir
#         stain.new_dir = args.new_dir
#         stain.predict()