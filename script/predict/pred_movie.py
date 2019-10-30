# coding=utf-8
import os
import sys
import cv2
import numpy as np
import torch
from script.model_make.models import resnet18_2D
root_dir = os.path.join(os.getcwd(), '../../')
sys.path.append(root_dir)


def main():
    # 0. config setting
    # target_movie_file = os.path.join('/usr', 'local', 'wk', 'data', 'fortnite', 'movie', '8_test.mp4')
    target_movie_file = '/usr/local/wk/data/fortnite/test/8_test.mp4'
    cap = cv2.VideoCapture(target_movie_file)
    org_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    org_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    org_total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    org_fps = cap.get(cv2.CAP_PROP_FPS)
    total_sec = int(org_total_frames / org_fps)
    pred_using_frame_num = 5
    pred_size = (800, 450)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = 2
    n_images = 5
    ngf = 64

    model_file = os.path.join(root_dir, 'script', 'model_make', 'saved', 'trained_weight_290.pth')
    model = resnet18_2D(num_classes, n_images * 3, ngf).to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    frame_cnt = 0
    pred_torch_image_list = []

    print('[INFO] movie_file_name:{}\n'
          '       width:{} height:{} fps:{} length:{}s'.format(
            target_movie_file, org_width, org_height, org_fps, total_sec))

    if not cap.isOpened():
        print('target movie file is not loaded.')
        exit(1)

    result_list = []

    """
    while True:
        ret, frame = cap.read()
        if ret:
            # frameが存在する場合
            frame_cnt += 1
            print('[INFO] frame_num:{}'.format(frame_cnt))
            # 1s 1fで画像データを取得
            if int(frame_cnt % org_fps) == 1:
                # 解析用のサイズにリサイズ
                frame = cv2.resize(frame, pred_size)
                frame = np.transpose(frame, (2, 0, 1)) / 255.  # BGRからRGBに
                tensor_img = torch.from_numpy(frame).float()
                pred_torch_image_list.append(tensor_img)

            if len(pred_torch_image_list) == pred_using_frame_num:
                print('[INFO] 5frame get! pred start.')
                data = torch.cat(pred_torch_image_list, 0)
                data = data.unsqueeze(0).to(device)
                valid_outputs = model(data)
                if valid_outputs[0][0] > valid_outputs[0][1]:
                    result = 0  # not in battle
                else:
                    result = 1  # in battle
                result_list.append(result)
                pred_torch_image_list.clear()
                print(result_list)

        else:
            # frameが存在しない場合
            print('[INFO] finish.')
            exit(2)
    """

    result_list = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1,
                   1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1,
                   1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                   1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                   1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0,
                   1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1,
                   1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                   1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]



if __name__ == '__main__':
    main()
