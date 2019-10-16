import os
import glob
import numpy as np
import dill
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from datasets import SliceDataset
from models import resnet10_2D, resnet18_2D
from pytorch_utils import model_outputs_with_ground_truth, \
    rand_init

# スライス毎の出血判別モデルに対する、CAMのヒートマップ作成モデルを構築する


def get_load_func(device):
    def load_func(data):
        x, y = data
        x = x.to(device)
        y = y.to(device)
        return x, y

    return load_func


def get_heat_maps(model, dataloader, w):
    """
    CAM(https://arxiv.org/abs/1512.04150)の計算
    """
    heat_map_all = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0]
            f = model.forward_until_avgpool(x.to(device))
            cam = torch.einsum("ijkl,j->ikl", (f, w)).detach().cpu().numpy()
            # ゼロ未満の部分をゼロに
            heat_map = np.maximum(cam, 0)
            # スキャンインスタンスごとにmaxで割って、(0, 1)に正規化
            heat_map /= np.amax(heat_map, axis=(1,2), keepdims=True)
            heat_map_all.append(heat_map)

    return np.concatenate(heat_map_all, axis=0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')

    # basic settings
    parser.add_argument("--seed", type=int, default=1,
                        help="")
    parser.add_argument("--save_dir", type=str,
                        default="../../models/cam_results/init",
                        help="CAMの結果を保存するディレクトリ")
    # dataset
    parser.add_argument("--data_root_dir", type=str,
                        default="../../data/interim/dcm_img_A0-B80-imgsize224_aligned/",
                        help="画像ファイルが含まれている大元のディレクトリ")
    parser.add_argument("--valid_csv_path", type=str,
                        default="../../data/processed/datasets_split0/valid.csv",
                        help="検証用データの各インスタンスのディレクトリ名、ラベルが記述されているcsvファイル")

    # preprocess, augmentation
    parser.add_argument('--augmentation', action='store_true',
                        help='学習時、入力画像に対してaugmentation(horizontal flip, random rotation, random crop)を行うかどうか')
    parser.add_argument('--scale_center_05', action='store_true',
                        help='Trueの時は(0, 1)に、Falseのときは(-1, 1)に画像をスケーリングする')

    # data loader
    parser.add_argument("--num_workers", type=int, default=4,
                        help="dataloaderの並列数")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="batch size")

    # model
    parser.add_argument("--num_classes", type=int, default=6,
                        help="何種類のマルチラベル分類を解くか。単純な出血判別モデルの場合は1とする")
    parser.add_argument("--in_channels", type=int, default=1,
                        help="入力画像のチャネル数")
    parser.add_argument("--model_type", type=str, default="resnet10_2D",
                        help="", choices=['resnet10_2D', 'resnet18_2D'])
    parser.add_argument("--n_slices", type=int, default=32,
                        help="ScanDatasetを用いるときの、入力に用いるスライスの枚数")
    parser.add_argument("--weights_dir", type=str,
                        default="../../models/weights/init",
                        help="重みを保存するディレクトリ")
    args = parser.parse_args()
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rand_init(args.seed)
    valid = SliceDataset(args.valid_csv_path,
                         args.data_root_dir,
                         training=False,
                         scale_center_05=args.scale_center_05,
                         augmentation=args.augmentation)
    

    valid_dataloader = DataLoader(valid, batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers)


    if args.model_type == 'resnet10_2D':
        # ここより下、2Dのモデルのため、入力のチャネル数がスライス枚数と一致
        model = resnet10_2D(args.num_classes,
                            args.n_slices).to(device)
    elif args.model_type == 'resnet18_2D':
        model = resnet18_2D(args.num_classes,
                            args.n_slices).to(device)

    print("model:", model)


    load_func = get_load_func(device)

    # weights_dir内の重みファイルをすべて取得
    weights_paths = sorted(glob.glob(args.weights_dir + "/*.pth"))
    os.makedirs(args.save_dir, exist_ok=True)

    # 各重みに対してCAMの結果を計算して保存する
    for weights_path in weights_paths:
        print(weights_path)
        weights = torch.load(weights_path)
        model.load_state_dict(weights)
        model.eval()
        w = model.fc.weight.data.squeeze()
        heat_map_arr = get_heat_maps(model, valid_dataloader, w)
        cam_save_path = os.path.join(args.save_dir, os.path.basename(weights_path).replace(".pth", "_cam"))
        np.save(cam_save_path, heat_map_arr, allow_pickle=True)
