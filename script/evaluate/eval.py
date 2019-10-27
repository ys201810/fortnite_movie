# coding=utf-8
import os
import sys
import cv2
import glob
import torch
import numpy as np
from script.model_make.models import resnet18_2D
root_dir = os.path.join(os.getcwd(), '../../')
sys.path.append(root_dir)


def main():
    # config setting
    data_dir = os.path.join('/usr', 'local', 'wk', 'data', 'fortnite')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    annotation_file = os.path.join(data_dir, 'annotation.list')

    num_classes = 2
    n_images = 5
    ngf = 64

    model_file = os.path.join(root_dir, 'script', 'model_make', 'saved', 'trained_weight_100.pth')
    model = resnet18_2D(num_classes, n_images * 3, ngf).to(device)

    model_dir = '/usr/local/wk/fortnite_movie/script/model_make/saved/'
    models = glob.glob(model_dir + '*.pth')
    models = sorted(models)
    for model_file in models:
        eval_num = 0
        ok_num = 0
        label_0_num = 0
        label_1_num = 0
        label_0_ok_num = 0
        label_1_ok_num = 0
        model.load_state_dict(torch.load(model_file))
        model.eval()
        with open(annotation_file, 'r') as inf:
            for line in inf:
                line = line.rstrip()
                vals = line.split(' ')
                image_paths = vals[0]
                kind = vals[1]
                if kind != 'val':
                    continue
                eval_num += 1
                label = int(vals[2])
                if label == 0:
                    label_0_num += 1
                else:
                    label_1_num += 1
                img_tensor_list = []
                image_files = glob.glob(image_paths)
                image_files = sorted(image_files)
                for image_file in image_files:
                    img = cv2.imread(image_file)
                    img = np.transpose(img, (2, 0, 1)) / 255.  # BGRからRGBに
                    tensor_img = torch.from_numpy(img).float()
                    img_tensor_list.append(tensor_img)

                imgs = torch.cat(img_tensor_list, 0)
                imgs = imgs.unsqueeze(0).to(device)
                valid_outputs = model(imgs)
                pred = 0 if valid_outputs[0][0] > valid_outputs[0][1] else 1
                if pred == int(label):
                    result = 'ok'
                    ok_num += 1
                    if label == 0:
                        label_0_ok_num += 1
                    else:
                        label_1_ok_num += 1
                else:
                    result = 'ng'
                # print(result, label, valid_outputs)

        print('model_file:{} total_correct_rate:{} not_battle_correct_rate:{} battle_correct_rate:{}'.format(
            model_file.split('/')[-1], round(ok_num / eval_num, 3),
            round(label_0_ok_num / label_0_num, 3),
            round(label_1_ok_num / label_1_num, 3)
        ))


if __name__ == '__main__':
    main()
