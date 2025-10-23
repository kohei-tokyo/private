from main_module import DDPM

if __name__ == '__main__':
    model = DDPM(
        dir=r"D:\Matsusaka\data_mito\HeLa_Su9-mSG_original_crop",
        name=f"Test",
        train_folders=["2", "3", "4"],
        val_folders=["1"],
        test_folders=["5"],
        n_epoch=2
    )
    model.all()