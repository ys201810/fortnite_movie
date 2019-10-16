
import os
import glob
import numpy as np
import dill
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from datasets import ScanDataset, SliceDataset
from models import resnet10, resnet18, SmallAlexNet, SmallAlexNet2, \
    resnet10_2D, resnet18_2D, SmallAlexNet_2D, SmallAlexNet2_2D
from pytorch_utils import model_outputs_with_ground_truth, \
    rand_init, BCELossWithLogits


# モデルの出力結果を算出してnumpy形式で保存するスクリプト
# モデルの出力にsigmoid変換は施していない


def get_load_func(device, feed_weight):
    if feed_weight:
        def load_func(data):
            x, y, _ = data
            x = x.to(device)
            y = y.to(device)
            return x, y
    else:
        def load_func(data):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            return x, y
        

    return load_func


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')

    # basic settings
    parser.add_argument("--seed", type=int, default=1,
                        help="")
    parser.add_argument("--save_dir", type=str,
                        default="../../models/predict_results/init",
                        help="モデルの予測結果を保存するディレクトリ")


    # dataset
    parser.add_argument("--data_root_dir", type=str,
                        default="../../data/interim/dcm_img_A0-B80-imgsize224_aligned/",
                        help="画像ファイルが含まれている大元のディレクトリ")
    parser.add_argument("--valid_csv_path", type=str,
                        default="../../data/processed/datasets_split0/valid.csv",
                        help="検証用データの各インスタンスのディレクトリ名、ラベルが記述されているcsvファイル")
    parser.add_argument("--n_slices", type=int, default=32,
                        help="ScanDatasetを用いるときの、入力に用いるスライスの枚数")
    parser.add_argument('--classif_hemorrhage_or_not', action='store_true',
                        help='出血しているかどうかだけの判別とするか')
    parser.add_argument('--slices_range', type=str, default='all', choices=['all', 'only_brain', 'brain_and_the_below'])
    parser.add_argument("--label_type", type=str, default="6types",
                        choices=['6types', 'ich', 'sah'],
                        help="SliceDatasetを用いた学習を行う場合の、正解ラベル。6種類全てか、ICHか、SAHか。")
    parser.add_argument("--confidence_col", type=str, default="",
                        help="損失関数の重み調整フラグを表す列")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="ある特定の(=confidence_colで指定した列の値が0)学習インスタンスに対する、損失関数の重みの値")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="SliceDatasetにおける、学習に用いるスライスの範囲を示すパラメータ(詳細はSliceDatasetで)。")
    parser.add_argument("--dataset", type=str, default="scan", choices=['scan', 'slice'],
                        help="SliceDataset or ScanDataset")

    # preprocess, augmentation
    parser.add_argument('--augmentation', action='store_true',
                        help='学習時、入力画像に対してaugmentation(horizontal flip, random rotation, random crop)を行うかどうか')
    parser.add_argument('--scale_center_05', action='store_true',
                        help='Trueの時は(0, 1)に、Falseのときは(-1, 1)に画像をスケーリングする')
    parser.add_argument('--brightness_aug', action='store_true',
                        help='学習時、入力画像に対して明るさに関するaugmentationを行うかどうか')
    parser.add_argument('--affine_aug', action='store_true',
                        help='学習時、入力画像に対して平行移動・せん断ひずみに関するaugmentationを行うかどうか')

    # data loader
    parser.add_argument("--num_workers", type=int, default=4,
                        help="dataloaderの並列数")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="batch size")

    # model
    parser.add_argument("--shortcut_type", type=str, default="A",
                        help="resnetのショートカットタイプ。modelでresnetを使用しているときのみ使用する")
    parser.add_argument("--num_classes", type=int, default=6,
                        help="何種類のマルチラベル分類を解くか。単純な出血判別モデルの場合は1とする")
    parser.add_argument("--in_channels", type=int, default=1,
                        help="入力画像のチャネル数")
    parser.add_argument("--ngf", type=int, default=64,
                        help="CNNの中間層におけるチャネル数の最小単位")
    parser.add_argument("--model_type", type=str, default="resnet10",
                        help="", choices=['resnet10', 'resnet18', 'smallalexnet', 'smallalexnet2',
                                          'resnet10_2D', 'resnet18_2D', 'smallalexnet_2D', 'smallalexnet2_2D'])
    parser.add_argument("--weights_dir", type=str,
                        default="../../models/weights/init",
                        help="重みが保存されているディレクトリ")
    
    args = parser.parse_args()
    feed_weight = args.confidence_col != ""
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rand_init(args.seed)

    # datasetの作成
    if args.dataset == 'scan':
        # 3Dモデルの場合は、loadしたデータに新たにチャネル方向の次元を加える
        expand = args.model_type in ['resnet10', 'resnet18', 'smallalexnet', 'smallalexnet2']
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
        valid = SliceDataset(args.valid_csv_path,
                             args.data_root_dir,
                             training=False,
                             scale_center_05=args.scale_center_05,
                             augmentation=args.augmentation,
                             brightness_aug=args.brightness_aug,
                             affine_aug=args.affine_aug,
                             confidence_col=args.confidence_col,
                             alpha=args.alpha)


    valid_dataloader = DataLoader(valid, batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers)

    
    if args.model_type == 'resnet10':
        print("resnet10を用いて予測します")
        model = resnet10(args.shortcut_type, args.num_classes,
                         args.in_channels).to(device)
    elif args.model_type == 'resnet18':
        print("resnet18を用いて予測します")
        model = resnet18(args.shortcut_type, args.num_classes,
                         args.in_channels).to(device)
    elif args.model_type == 'smallalexnet':
        print("SmallAlexNetを用いて予測します")
        model = SmallAlexNet(num_classes=args.num_classes,
                             in_channels=args.in_channels).to(device)
    elif args.model_type == 'smallalexnet2':
        print("SmallAlexNet2を用いて予測します")
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
        print("SmallAlexNet_2Dを用いて予測します")
        model = SmallAlexNet_2D(num_classes=args.num_classes,
                                in_channels=args.n_slices).to(device)
    elif args.model_type == 'smallalexnet2_2D':
        print("SmallAlexNet2_2Dを用いて予測します")
        model = SmallAlexNet2_2D(num_classes=args.num_classes,
                                 in_channels=args.n_slices).to(device)
    else:
        raise ValueError("不適切なmodel_typeです")

    print("model:", model)

    load_func = get_load_func(device, feed_weight)

    # weights_dir内の重みファイルをすべて取得
    weights_paths = sorted(glob.glob(args.weights_dir + "/*.pth"))
    os.makedirs(args.save_dir, exist_ok=True)

    # 各重みに対して予測結果を計算して保存する
    for weights_path in weights_paths:
        print(weights_path)
        weights = torch.load(weights_path)
        model.load_state_dict(weights)
        y_pred, y_true = model_outputs_with_ground_truth(model, valid_dataloader, load_func=load_func)
        pred_save_path = os.path.join(args.save_dir, os.path.basename(weights_path).replace(".pth", "_pred"))
        np.save(pred_save_path, y_pred, allow_pickle=True)
        true_save_path = os.path.join(args.save_dir, os.path.basename(weights_path).replace(".pth", "_true"))
        np.save(true_save_path, y_true, allow_pickle=True)
