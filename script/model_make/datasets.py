# coding=utf-8
import cv2
import glob
import torch
import numpy as np
from torch.utils.data import Dataset


class FiveSecDataset(Dataset):
    def __init__(self, annotation_file, target, inputs_transform):
        self.data = []
        self.label = []
        self.inputs_transform = inputs_transform
        with open(annotation_file, 'r') as inf:
            for i, line in enumerate(inf):
                print(i)
                line = line.rstrip()
                vals = line.split(' ')
                image_paths = vals[0]
                kind = vals[1]
                label = int(vals[2])
                if kind == target:
                    image_files = glob.glob(image_paths)
                    image_files = sorted(image_files)
                    self.data.append(image_files)
                    self.label.append(label)
            self.data_num = len(self.data)

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        image_files = self.data[index]
        img_tensor_list = []

        for image_file in image_files:
            img = cv2.imread(image_file)
            img = np.transpose(img, (2, 0, 1)) / 255.  # BGRからRGBに
            tensor_img = torch.from_numpy(img).float()
            img_tensor_list.append(tensor_img)

        out_data = torch.cat(img_tensor_list, 0)

        out_label = torch.from_numpy(np.array(self.label[index]))
        out_label = torch.eye(2)[out_label]

        return out_data, out_label


def main():
    annotation_file = '/usr/local/wk/data/fortnite/annotation.list'
    FiveSecDataset(annotation_file, 'train', None)


if __name__ == '__main__':
    main()

