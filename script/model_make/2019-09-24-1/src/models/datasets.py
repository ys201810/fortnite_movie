import os
import warnings
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy.ndimage.interpolation import rotate
import cv2


def quotedval2val(input_):
    """
    dicomファイルのタグに指定されてい文字列からクォーテーションを取り除いてfloatで返す
    例えば、
    input_='5.00000'
    であった場合、この関数の返り値は
    5.0
    となる
    """
    res = ""
    try:
        for c in input_:
            if c != "\'" and c != '\"':
                res += c

        return np.float32(float(res))
    except TypeError:
        return -1


class ScanDataset(Dataset):
    def __init__(self, csv_path, root_dir, training, n_slices=32,
                 classif_hemorrhage_or_not=True,
                 label_type='6types',
                 confidence_col="",
                 alpha=1.0,
                 beta=0.0, 
                 augmentation=False,
                 scale_center_05=False,
                 slices_range='all',
                 expand=True):
        """
        * csv_path:各患者の出血情報が記載されたcsvファイルのパス
        * root_dir:各患者のスキャンデータを含んでいるディレクトリを含まれているディレクトリ
        * slices_range:スライスの使用する部分。
        * label_type : 6typesなら、ICH~SAHの6種類をラベルを使う
                       ichなら、3人のICHの結果を用いる
        * confidence_col : dfにおける、重みをつけるかどうかのフラグ変数の列名
                       confidence_colで指定された列の値がゼロならば、重みをつける対象
                       ""ならば、すべてのインスタンスで重みは1。
        * alpha : 重みの値
        * beta : 脳の開始からどれくらい下までをモデルの入力とするか。打ち合わせでは、4/14程が良いのでは、という話になった。
                 beta=0だと、脳の開始から脳の終了までをみることになる
        * expand_dimsして4D tensorにするかどうか
        """
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.n_slices = n_slices
        self.training = training
        self.classif_hemorrhage_or_not = classif_hemorrhage_or_not
        self.augmentation = augmentation
        self.scale_center_05 = scale_center_05
        if label_type in ["ich", "sah"]:
            warnings.warn("classif_hemorrhage_or_notは無視されます")
        self.label_type = label_type
        self.confidence_col = confidence_col
        self.alpha = alpha
        self.beta = beta
        self.slices_range = slices_range
        self.expand = expand

    def _get_slices(self, slice_paths, slice_thickness):
        """
        学習時にはパディングや読み込み開始位置をランダムに選ぶ
        """
        if slice_thickness == 5:
            n_skip_slices = 1
        elif slice_thickness == 1:
            n_skip_slices = 5
        elif slice_thickness == 0.625:
            n_skip_slices = 8
        else:
            raise ValueError("不適切なslice_thicknessです")

        slices = []
        # n_residual:そのスキャンに於いて余るスライスの枚数
        n_residual = len(slice_paths) - n_skip_slices * self.n_slices
        if n_residual >= 0:
            # スライス枚数が足りている時
            # ランダムにスタート位置を振る
            start_index = np.random.randint(n_residual + 1) if self.training else n_residual // 2
            for slice_i, slice_path in enumerate(slice_paths):
                if (slice_i >= start_index) and (slice_i % n_skip_slices == 0):
                    slice = np.asarray(Image.open(slice_path))
                    slices.append(slice)

                if len(slices) == self.n_slices:
                    break

            assert len(slices) == self.n_slices
        else:
            # スライス枚数が足りていない時
            # 開始スライスをランダムにふる
            for slice_i, slice_path in enumerate(slice_paths[np.random.randint(n_skip_slices):]):
                if (slice_i % n_skip_slices == 0):
                    slice = np.asarray(Image.open(slice_path))
                    slices.append(slice)

            n_shortage = self.n_slices - len(slices)
            n_head = np.random.randint(n_shortage + 1) if self.training else n_shortage // 2
            n_tail = n_shortage - n_head
            pad_slices_head = [np.zeros_like(slices[0]) for _ in range(n_head)]
            pad_slices_tail = [np.zeros_like(slices[0]) for _ in range(n_tail)]
            slices = pad_slices_head + slices + pad_slices_tail
            assert len(slices) == self.n_slices

        return slices

    def __getitem__(self, index):
        target_record = self.df.iloc[index]
        patient_id = target_record['id']
        subdirname = target_record['dir_name']
        slice_thickness = quotedval2val(target_record['Slice Thickness'])
        scan_dir = os.path.join(self.root_dir,
                                patient_id + " " + patient_id,
                                "Unknown Study",
                                subdirname)
        # get slices
        slice_paths = sorted(glob.glob(scan_dir + '/*.png'))
        if self.slices_range == 'only_brain':
            # 脳が写っている部分だけを用いる
            # アノテーション時のスライス番号は1から、梅垣がつけたスライス番号は0から割り振っているので
            # -1する
            slice_paths = slice_paths[target_record['脳開始スライス番号']-1:target_record['脳終了スライス番号']]
        elif self.slices_range == 'brain_and_the_below':
            # 脳の少し下まで。
            # 打ち合わせでの確認結果、脳の部分のスライス枚数のおおよそ4/14ほどまで下までにみたらよいということになった。
            # もしかしたらもう少し下まで見たほうがいいかもしれないので、ここはパラメータとする
            _n_brain_slices = target_record['脳終了スライス番号'] - target_record['脳開始スライス番号'] + 1
            _n_brain_and_below_slices = int(round(_n_brain_slices * (1 + self.beta)))
            # インデックスが負にならないよう、0とのmaxをとる
            _start_slice_index = max(target_record['脳終了スライス番号'] - _n_brain_and_below_slices, 0)
            slice_paths = slice_paths[_start_slice_index:target_record['脳終了スライス番号']]
        elif self.slices_range == 'all':
            # 全部使用。スルー。
            pass
        else:
            raise ValueError("不適切なslices_range")
        slices = self._get_slices(slice_paths, slice_thickness)
        slices = np.array(slices)
        if self.training and self.augmentation:
            # data augmentation
            # Horizontal Flip
            if np.random.rand() < 0.5:
                slices = slices[:, :, ::-1]

            # random rotation
            angle = np.random.randint(-10, 10 + 1)
            _, h, w = slices.shape
            slices = cv2.resize(rotate(slices.transpose(1, 2, 0), angle), (h, w)).transpose(2, 0, 1)

            # random crop
            r = 0.9
            h2 = int(h * r)
            w2 = int(w * r)
            start_h = np.random.randint(h - h2)
            start_w = np.random.randint(w - w2)
            slices = slices[:, start_h:(start_h + h2), start_w:(start_w + w2)]
            slices = cv2.resize(slices.transpose(1, 2, 0), (h, w)).transpose(2, 0, 1)
        elif not self.training and self.augmentation:
            # augmentation時にrandom cropしている分スキャンが大きく写ってしまっているので、
            # 予測・評価時にも大きさを揃える
            _, h, w = slices.shape
            r = 0.9
            h2 = int(h * r)
            w2 = int(w * r)
            start_h = (h - h2) // 2
            start_w = (w - w2) // 2
            slices = slices[:, start_h:(start_h + h2), start_w:(start_w + w2)]
            slices = cv2.resize(slices.transpose(1, 2, 0), (h, w)).transpose(2, 0, 1)

        if self.expand:
            slices = np.expand_dims(slices, 0)     # (1, 32, 224, 224)


        # 入力データの正規化の方法を決める
        if self.scale_center_05:
            slices = slices / 255  # (0, 1)にスケーリング
        else:
            slices = (slices / 255) * 2 - 1  # (-1, 1)にスケーリング



        # 正解データ(ラベル)を決める
        if self.label_type == "6types":
            # get label
            label = target_record[['ICH', 'IPH', 'IVH', 'SDH', 'EDH', 'SAH']].values.astype(int)
            if self.classif_hemorrhage_or_not:
                # logical or を取る
                label = np.max(label)
        elif self.label_type == "ich":
            label = target_record['ich_label']
        elif self.label_type == "sah":
            label = target_record['sah_label']
        else:
            raise ValueError("不適切なフラグの値")

        label = np.expand_dims(label, 0) 


        # 学習インスタンスに重みをfeedするかどうかを決める
        if self.confidence_col:
            # 重みを使う時
            if target_record[self.confidence_col] == 1:
                # フラグが1なので重み付けしない
                weight = 1.0
            elif target_record[self.confidence_col] == 0:
                # 重みをつける
                weight = self.alpha
            else:
                raise ValueError("不適切なフラグの値")

            weight = np.expand_dims(weight, 0)

            return torch.tensor(slices, dtype=torch.float32), torch.tensor(label, dtype=torch.float32),\
                torch.tensor(weight, dtype=torch.float32)
        else:
            # 重みを使わない時
            return torch.tensor(slices, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.df)


class SliceDataset(Dataset):
    def __init__(self, csv_path, root_dir, training, 
                 augmentation=False,
                 scale_center_05=False,
                 brightness_aug=False,
                 affine_aug=False,
                 alpha=1.0, 
                 confidence_col="",
                 csv_path_mizumashi="",
                 n_mizumashi=1):
        """
        * csv_path:各患者の出血情報が記載されたcsvファイルのパス
        * root_dir:各患者のスキャンデータを含んでいるディレクトリを含まれているディレクトリ
        * brightness_aug:画像の明るさに関するaugmentation
        * affine_aug:画像のアフィン変換(水平移動・せん断ひずみ):に関するaugmentation
        * confidence_col : dfにおける、重みをつけるかどうかのフラグ変数の列名
                       confidence_colで指定された列の値がゼロならば、重みをつける対象
                       ""ならば、すべてのインスタンスで重みは1。
        * alpha : 重みの値
        """
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.training = training
        self.augmentation = augmentation
        self.scale_center_05 = scale_center_05
        self.confidence_col = confidence_col
        self.alpha = alpha

        # create transformer of image
        _transform_list = []
        if brightness_aug:
            warnings.warn("明るさに関するaugmentationを施します")
            transform = transforms.ColorJitter(brightness=0.5)
            _transform_list.append(transform)

        if affine_aug:
            warnings.warn("augmentationとして、画像に水平移動とせん断ひずみを施します")
            transform = transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=None, shear=10, resample=False, fillcolor=0)
            _transform_list.append(transform)

        self.transform = transforms.Compose(_transform_list)


        # 特定の学習インスタンスの水増し
        if csv_path_mizumashi and n_mizumashi >= 2:
            warnings.warn("学習データの一部の水増しを行います")
            # 水増しする学習データ取得
            df_mizumashi = pd.read_csv(csv_path_mizumashi)
            # 水増しされるスライスにおける出血ラベルや信頼度の情報を得る
            df_mizumashi = df_mizumashi.merge(self.df)
            # 追加分のレコードを作成
            # 2倍に水増しする際にはdf_mizumashi_2を1個作成してそれをそれをdf_trainにくっつける
            df_mizumashi = pd.concat([df_mizumashi for _ in range(n_mizumashi - 1)])
            # df_trainにくっつける
            self.df = pd.concat([self.df, df_mizumashi]).reset_index(drop=True)
        else:
            warnings.warn("特定のデータの水増しは行いません。")

    def __getitem__(self, index):
        target_record = self.df.iloc[index]
        patient_id = target_record['id']
        subdirname = target_record['dir_name']
        slice_no = target_record['slice_no']
        scan_dir = os.path.join(self.root_dir,
                                patient_id + " " + patient_id,
                                "Unknown Study",
                                subdirname)
        # get slice
        # dicomビューアはスライスを1からカウントしているが、
        # 梅垣はスライスを0からカウントしているため、-1する
        slice_path = os.path.join(scan_dir, "ALIGNED{0:06d}.png".format(slice_no - 1))

        # 画像self.trainingの値に関わらず、はtransformして読み込む
        slice = np.asarray(self.transform(Image.open(slice_path)))
        if self.training and self.augmentation:
            # data augmentation
            # Horizontal Flip
            if np.random.rand() < 0.5:
                slice = slice[:, ::-1]

            # random rotation
            angle = np.random.randint(-10, 10 + 1)
            h, w = slice.shape
            slice = cv2.resize(rotate(slice, angle), (h, w))

            # random crop
            r = 0.9
            h2 = int(h * r)
            w2 = int(w * r)
            start_h = np.random.randint(h - h2)
            start_w = np.random.randint(w - w2)
            slice = slice[start_h:(start_h + h2), start_w:(start_w + w2)]
            slice = cv2.resize(slice, (h, w))
        elif not self.training and self.augmentation:
            # augmentation時にrandom cropしている分スキャンが大きく写ってしまっているので、
            # 予測・評価時にも大きさを揃える
            h, w = slice.shape
            r = 0.9
            h2 = int(h * r)
            w2 = int(w * r)
            start_h = (h - h2) // 2
            start_w = (w - w2) // 2
            slice = slice[start_h:(start_h + h2), start_w:(start_w + w2)]
            slice = cv2.resize(slice, (h, w))

        # 入力データの正規化の方法を決める
        if self.scale_center_05:
            slice = slice / 255  # (0, 1)にスケーリング
        else:
            slice = (slice / 255) * 2 - 1  # (-1, 1)にスケーリング


        # channelとして新たな次元を加える
        slice = np.expand_dims(slice, 0)

        # 正解データ(ラベル)を決める       
        label = np.expand_dims(target_record['出血フラグ'], 0) 


        # 学習インスタンスに重みをfeedするかどうかを決める
        if self.confidence_col:
            # 重みを使う時
            if target_record[self.confidence_col] == 1:
                # フラグが1なので重み付けしない
                weight = 1.0
            elif target_record[self.confidence_col] == 0:
                # 重みをつける
                weight = self.alpha
            else:
                raise ValueError("不適切なフラグの値")

            weight = np.expand_dims(weight, 0)

            return torch.tensor(slice, dtype=torch.float32), \
                torch.tensor(label, dtype=torch.float32),\
                torch.tensor(weight, dtype=torch.float32)
        else:
            return torch.tensor(slice, dtype=torch.float32), \
                torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.df)


