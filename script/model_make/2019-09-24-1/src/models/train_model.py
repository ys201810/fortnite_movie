
import os
import sys
import warnings
import dill
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from datasets import ScanDataset, SliceDataset
from models import resnet10, resnet18, SmallAlexNet, SmallAlexNet2, \
    resnet10_2D, resnet18_2D, SmallAlexNet_2D, SmallAlexNet2_2D
from pytorch_utils import worker_init_fn, create_dataset, rand_init, BasicTrainerBase, mixup_data
from utils import save_src_files

# モデルの学習を行うスクリプト


def create_save_weights_func(dir_):
    """
    ディレクトリ名を引数にして、「そのディレクトリにモデルの重みを保存する関数」を返す関数
    """
    def save_weights(model, epoch):
        torch.save(model.state_dict(), os.path.join(dir_, "epoch{0:04d}.pth".format(epoch + 1)))
    return save_weights


class Trainer(BasicTrainerBase):
    def __init__(self, model, optimizer, device,
                 scheduler, scheduler_plateau, feed_weight,
                 mixup_alpha):
        """
        * feed_weight : 学習時に学習インスタンスに重みをつけるかどうか
        """
        super().__init__(model, optimizer, device)
        self.scheduler = scheduler
        self.scheduler_plateau = scheduler_plateau
        self.feed_weight = feed_weight
        self.mixup_alpha = mixup_alpha
        if self.mixup_alpha > 0:
            warnings.warn("データのmixupを行います")

    def loss_forward(self, data):
        if self.feed_weight:
            # 重みありの誤差関数
            x, y, w = data
            x = x.to(device)
            y = y.to(device)
            x, y = mixup_data(x, y, device, alpha=self.mixup_alpha)
            w = w.to(device)
            criterion = nn.BCEWithLogitsLoss(weight=w)
            outputs = self.model(x)
            loss = criterion(outputs, y)
        else:
            # 重みなしの誤差関数
            x, y = data
            x = x.to(device)
            y = y.to(device)
            x, y = mixup_data(x, y, device, alpha=self.mixup_alpha)
            criterion = nn.BCEWithLogitsLoss()
            outputs = self.model(x)
            loss = criterion(outputs, y)

        return loss
        

    def after_each_epoch(self):
        if self.scheduler is not None:
            if self.scheduler_plateau:
                self.scheduler.step(self.valid_costs_lst[-1])
            else:
                self.scheduler.step()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')

    # basic settings
    parser.add_argument("--seed", type=int, default=1,
                        help="")

    # dataset
    parser.add_argument("--data_root_dir", type=str,
                        default="../../data/interim/dcm_img_A0-B80-imgsize224_aligned/",
                        help="画像ファイルが含まれている大元のディレクトリ")
    parser.add_argument("--train_csv_path", type=str,
                        default="../../data/processed/datasets_split0/train.csv",
                        help="学習用データの各インスタンスのディレクトリ名、ラベルが記述されているcsvファイル")
    parser.add_argument("--valid_csv_path", type=str,
                        default="../../data/processed/datasets_split0/valid.csv",
                        help="検証用データの各インスタンスのディレクトリ名、ラベルが記述されているcsvファイル")
    parser.add_argument("--n_slices", type=int, default=32,
                        help="ScanDatasetを用いるときの、入力に用いるスライスの枚数")
    parser.add_argument('--classif_hemorrhage_or_not', action='store_true',
                        help='出血しているかどうかだけの判別とするか')
    parser.add_argument('--slices_range', type=str, default='all', choices=['all', 'only_brain', 'brain_and_the_below'],
                        help='ScanDatasetを用いるときの、入力に用いるスライスの範囲')
    parser.add_argument("--confidence_col", type=str, default="",
                        help="損失関数の重み調整フラグを表す列")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="ある特定の(=confidence_colで指定した列の値が0)学習インスタンスに対する、損失関数の重みの値")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="SliceDatasetにおける、学習に用いるスライスの範囲を示すパラメータ(詳細はSliceDatasetで)。")
    parser.add_argument("--dataset", type=str, default="scan", choices=['scan', 'slice'],
                        help="SliceDataset or ScanDataset")
    parser.add_argument("--label_type", type=str, default="6types",
                        choices=['6types', 'ich', 'sah'],
                        help="SliceDatasetを用いた学習を行う場合の、正解ラベル。6種類全てか、ICHか、SAHか。")

    # preprocess, augmentation
    parser.add_argument('--augmentation', action='store_true',
                        help='学習時、入力画像に対してaugmentation(horizontal flip, random rotation, random crop)を行うかどうか')
    parser.add_argument('--scale_center_05', action='store_true',
                        help='Trueの時は(0, 1)に、Falseのときは(-1, 1)に画像をスケーリングする')
    parser.add_argument('--brightness_aug', action='store_true',
                        help='学習時、入力画像に対して明るさに関するaugmentationを行うかどうか')
    parser.add_argument('--affine_aug', action='store_true',
                        help='学習時、入力画像に対して平行移動・せん断ひずみに関するaugmentationを行うかどうか')
    parser.add_argument("--mixup_alpha", type=float, default=0.0,
                        help="mixupを行う際のBeta分布のパラメータalphaの値。0.0の時、mixupは行われない")
    parser.add_argument("--csv_path_mizumashi", type=str, default="",
                        help="水増しする学習インスタンスを記述したcsvファイル")
    parser.add_argument("--n_mizumashi", type=int, default=1, help="1の場合、水増しは行わない")

    # data loader
    parser.add_argument("--num_workers", type=int, default=4,
                        help="dataloaderの並列数")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="batch size")

    # model
    parser.add_argument("--model_type", type=str, default="resnet10",
                        help="", choices=['resnet10', 'resnet18', 'smallalexnet', 'smallalexnet2',
                                          'resnet10_2D', 'resnet18_2D', 'smallalexnet_2D', 'smallalexnet2_2D'])
    parser.add_argument("--num_classes", type=int, default=6,
                        help="何種類のマルチラベル分類を解くか。単純な出血判別モデルの場合は1とする")
    parser.add_argument("--in_channels", type=int, default=1,
                        help="入力画像のチャネル数")
    parser.add_argument("--ngf", type=int, default=64,
                        help="CNNの中間層におけるチャネル数の最小単位")
    parser.add_argument("--shortcut_type", type=str, default="A",
                        help="resnetのショートカットタイプ。modelでresnetを使用しているときのみ使用する")

    # training
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--optimizer_type', type=str, default='Adam',
                        choices=['Adam', 'SGD'])
    parser.add_argument('--scheduler_step_if_sgd',
                        type=int, default= -1,
                        help="SGDの場合における、学習率を1/10にする頻度。-1にすると、ReduceLROnPlateauになる。Adam時には無視されるので注意")
    parser.add_argument('--weight_decay', type=float,
                        default=0.0, help="")
    parser.add_argument("--save_weights_interval", type=int,
                        default=1,
                        help="重みを保存する頻度。5の場合、5エポックに1回保存する")
    parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--save_weights_dir", type=str,
                        default="../../models/weights/init",
                        help="重みを保存するディレクトリ")

    args = parser.parse_args()
    # 重みをつけるかどうかは、信頼フラグの列を指定していかどうかで決める
    feed_weight = args.confidence_col != ""
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rand_init(args.seed)

    # datasetの作成
    if args.dataset == 'scan':
        # 3Dモデルの場合は、loadしたデータに新たにチャネル方向の次元を加える
        expand = args.model_type in ['resnet10', 'resnet18', 'smallalexnet', 'smallalexnet2']
        train = ScanDataset(args.train_csv_path,
                            args.data_root_dir,
                            training=True,
                            n_slices=args.n_slices,
                            classif_hemorrhage_or_not=args.classif_hemorrhage_or_not,
                            augmentation=args.augmentation,
                            scale_center_05=args.scale_center_05,
                            slices_range=args.slices_range, 
                            label_type=args.label_type,
                            confidence_col=args.confidence_col,
                            alpha=args.alpha,
                            beta=args.beta,
                            expand=expand)
        valid = ScanDataset(args.valid_csv_path,
                            args.data_root_dir,
                            training=False,
                            n_slices=args.n_slices,
                            classif_hemorrhage_or_not=args.classif_hemorrhage_or_not,
                            scale_center_05=args.scale_center_05,
                            augmentation=args.augmentation,
                            slices_range=args.slices_range, 
                            label_type=args.label_type,
                            confidence_col=args.confidence_col,
                            alpha=args.alpha,
                            beta=args.beta,
                            expand=expand)
    elif args.dataset == 'slice':
        warnings.warn("SliceDatasetを用います")
        train = SliceDataset(args.train_csv_path,
                             args.data_root_dir,
                             training=True,
                             augmentation=args.augmentation,
                             scale_center_05=args.scale_center_05,
                             brightness_aug=args.brightness_aug,
                             affine_aug=args.affine_aug,
                             alpha=args.alpha,
                             confidence_col=args.confidence_col,
                             csv_path_mizumashi=args.csv_path_mizumashi,
                             n_mizumashi=args.n_mizumashi)
        # 検証用データに関しては水増しを行わない
        valid = SliceDataset(args.valid_csv_path,
                             args.data_root_dir,
                             training=False,
                             scale_center_05=args.scale_center_05,
                             augmentation=args.augmentation,
                             brightness_aug=args.brightness_aug,
                             affine_aug=args.affine_aug,
                             alpha=args.alpha,
                             confidence_col=args.confidence_col)


    train_dataloader = DataLoader(train, batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  worker_init_fn=worker_init_fn)
    valid_dataloader = DataLoader(valid, batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  worker_init_fn=worker_init_fn)



    
    if args.model_type == 'resnet10':
        print("resnet10を学習させます")
        model = resnet10(args.shortcut_type, args.num_classes,
                         args.in_channels).to(device)
    elif args.model_type == 'resnet18':
        print("resnet18を学習させます")
        model = resnet18(args.shortcut_type, args.num_classes,
                         args.in_channels).to(device)
    elif args.model_type == 'smallalexnet':
        print("SmallAlexNetを学習させます")
        model = SmallAlexNet(num_classes=args.num_classes,
                             in_channels=args.in_channels).to(device)
    elif args.model_type == 'smallalexnet2':
        print("SmallAlexNet2を学習させます")
        model = SmallAlexNet2(num_classes=args.num_classes,
                              in_channels=args.in_channels).to(device)
    elif args.model_type == 'resnet10_2D':
        # ここより下、2Dのモデルのため、入力のチャネル数がスライス枚数と一致
        model = resnet10_2D(args.num_classes,
                            args.n_slices, ngf=args.ngf).to(device)
    elif args.model_type == 'resnet18_2D':
        model = resnet18_2D(args.num_classes,
                            args.n_slices, ngf=args.ngf).to(device)
    elif args.model_type == 'smallalexnet_2D':
        print("SmallAlexNet_2Dを学習させます")
        model = SmallAlexNet_2D(num_classes=args.num_classes,
                                in_channels=args.n_slices).to(device)
    elif args.model_type == 'smallalexnet2_2D':
        print("SmallAlexNet2_2Dを学習させます")
        model = SmallAlexNet2_2D(num_classes=args.num_classes,
                                 in_channels=args.n_slices).to(device)
    else:
        raise ValueError("不適切なmodel_typeです")

    print("model:", model)

    if args.optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
        scheduler = None
        scheduler_plateau = False
    elif args.optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                              nesterov=True, 
                              weight_decay=args.weight_decay)
        if args.scheduler_step_if_sgd == -1:
            warnings.warn("平坦になったタイミングで学習率を減衰させます")
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
            scheduler_plateau = True
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=args.scheduler_step_if_sgd,
                                                  gamma=0.1)
            scheduler_plateau = False
    else:
        raise ValueError("不適切なオプティマイザ")

    save_weights = create_save_weights_func(args.save_weights_dir)
    os.makedirs(args.save_weights_dir, exist_ok=True)
    callbacks_dct_using_model = {args.save_weights_interval:[save_weights]}
    trainer = Trainer(model, optimizer, device,
                      scheduler,
                      scheduler_plateau,
                      feed_weight,
                      args.mixup_alpha)
    history = trainer.training(train_dataloader, valid_dataloader,
                               n_epochs=args.n_epochs, 
                               callbacks_dct_using_model=callbacks_dct_using_model)
    print(history)
    with open(os.path.join(args.save_weights_dir, "history.pkl"), mode='wb') as f:
        dill.dump(history, f)

    # 学習に用いたスクリプトを保存
    save_src_files(os.path.abspath(os.path.dirname(__file__)), args.save_weights_dir)
