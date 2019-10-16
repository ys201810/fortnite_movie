import os
import sys
import dill
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import cv2
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"))
from datasets import SliceDataset
from pytorch_utils import worker_init_fn



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')

    # basic settings
    parser.add_argument("--cam_result_path", type=str,
                        default="default",
                        help="CAMの計算結果が保存されているディレクトリ")
    parser.add_argument("--save_dir", type=str, default="default",
                        help="CAMの可視化結果を保存するディレクトリ")

    # dataset
    parser.add_argument("--data_root_dir", type=str,
                        default="../../data/interim/dcm_img_A0-B80-imgsize224_aligned/",
                        help="画像ファイルが含まれている大元のディレクトリ")
    parser.add_argument("--valid_csv_path", type=str,
                        default="../../data/processed/datasets_split0/valid.csv",
                        help="検証用データの各インスタンスのディレクトリ名、ラベルが記述されているcsvファイル")    
    parser.add_argument("--n_slices", type=int, default=32,
                        help="ScanDatasetを用いるときの、入力に用いるスライスの枚数")

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
    args = parser.parse_args()
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid = SliceDataset(args.valid_csv_path,
                         args.data_root_dir,
                         training=False,
                         scale_center_05=args.scale_center_05,
                         augmentation=args.augmentation)

    valid_dataloader = DataLoader(valid, batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  worker_init_fn=worker_init_fn)

    heatmaps = np.load(args.cam_result_path)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    for (x, y), (_, v), heatmap in zip(valid, valid.df.iterrows(), heatmaps):
        if not args.scale_center_05:
            # 画像が(-1, 1)に正規化されているため、(0, 1)に戻す
            scan = np.uint8(255 * ((x.squeeze().numpy() + 1) / 2))
        # heatmapを(224, 224)にリサイズ
        img_size = 224
        heatmap = cv2.resize(heatmap, (img_size, img_size))
        # ヒートマップのスケールを(0, 1)から(0, 255)に変更
        heatmap = np.uint8(255 * heatmap)
        # ヒートマップの値を虹色のカラーマップで表現
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # スキャン画像とヒートマップ画像を重ね合わせる(0.8はheatmapの強調度合いを表すパラメータ)
        canvas = np.expand_dims(scan, -1) + 0.8 * heatmap
        # 得られた画像を再度(0, 255)に正規化
        canvas = np.uint8(canvas / canvas.max() * 255)

        # 画像を保存
        plt.figure(figsize=(3, 3))
        plt.imshow(canvas)
        plt.grid(False)
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        plt.tick_params(color='white')
        filename = (v['id'] + "__" + v['dir_name'] + "__" + str(v['slice_no']) + ".png").replace(" ", "_")
        print(filename)
        plt.savefig(os.path.join(save_dir, filename))
        plt.show()
