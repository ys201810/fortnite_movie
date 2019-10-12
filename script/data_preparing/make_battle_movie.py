# coding=utf-8
import os
import sys
import cv2
import glob
import copy
from script.data_preparing.utils import change_from_hms_to_sec, parse_battle_csv
root_dir = os.path.join(os.getcwd(), '../../')
sys.path.append(root_dir)


def make_from_anno(anno_csv, image_dir, movie_dir, fourcc):
    with open(anno_csv, 'r') as inf:
        for i, line in enumerate(inf):
            if i == 0 or i == 1 or i == 2:
                continue  # header
            line = line.rstrip()
            vals = line.split(',')
            no, name, length, url, battle_time, battle_timings = parse_battle_csv(line)
            if no != '8':
                continue
            for j, battle_timing in enumerate(battle_timings):
                start_time, end_time = battle_timing.split('~')
                start_times = start_time.split(':')
                start_sec = change_from_hms_to_sec(start_times)
                end_times = end_time.split(':')
                end_sec = change_from_hms_to_sec(end_times)
                print('player_name:{} battle_count:{} battle:{}~{}[sec]'.format(name, (j + 1), start_sec, end_sec))
                battle_images = []
                target_image_dir = os.path.join(image_dir, no)
                if end_sec < start_sec:
                    print('end sec smaller than start sec')
                    print('player_name:{} battle_count:{} battle:{}~{}[sec]'.format(name, (j + 1), start_sec, end_sec))
                    exit(1)
                for sec in range(start_sec, end_sec + 1):
                    battle_image_1 = glob.glob(os.path.join(target_image_dir, no + '_' + str(sec) + '_*.jpg'))[0]
                    # battle_images = battle_images + battle_images_tmp
                    battle_images.append(battle_image_1)

                first_img = cv2.imread(battle_images[0])
                height, width = first_img.shape[:2]
                video_file = os.path.join(movie_dir, no, '{}_{}_{}_{}.mp4'.format(no, str(j + 1), start_sec, end_sec))
                video = cv2.VideoWriter(video_file, fourcc, 5.0, (width, height))

                for battle_image in battle_images:
                    img = cv2.imread(battle_image)
                    video.write(img)

                video.release()


def make_from_dir(data_dir, fourcc, annotation_file):
    battle_img_list = []
    not_battle_img_list = []
    j = 0
    with open(annotation_file, 'r') as inf:
        for i, line in enumerate(inf):
            if i < 1333:
                continue
            j += 1
            line = line.rstrip()
            vals = line.split(' ')
            image_names = vals[0]
            label = vals[2]
            if label == '1':
                battle_img_list.append(image_names)
            else:
                not_battle_img_list.append(image_names)

            if j == 60:
                break

    battle_video_file = os.path.join(data_dir, 'battle_sample.mp4')
    print(len(battle_img_list), len(not_battle_img_list))

    battle_img_sample_list = []
    height, width = 360, 640 # first_img.shape[:2]
    for battle_imgs in battle_img_list:
        images = glob.glob(battle_imgs)
        images.sort()
        images_tmp = copy.copy(images)
        battle_img_sample_list = battle_img_sample_list + images_tmp
        # first_img = cv2.imread(images[0])

    video = cv2.VideoWriter(battle_video_file, fourcc, 5.0, (width, height))
    for image in battle_img_sample_list:
        img = cv2.imread(image)
        img = cv2.resize(img, (width, height))
        video.write(img)
    video.release()

    not_battle_video_file = os.path.join(data_dir, 'not_battle_sample.mp4')
    not_battle_imgs_sample_list = []
    for not_battle_imgs in not_battle_img_list:
        images = glob.glob(not_battle_imgs)
        images.sort()
        images_tmp = copy.copy(images)
        not_battle_imgs_sample_list = not_battle_imgs_sample_list + images_tmp
        # first_img = cv2.imread(images[0])
        # height, width = first_img.shape[:2]

    video = cv2.VideoWriter(not_battle_video_file, fourcc, 5.0, (width, height))
    for image in not_battle_imgs_sample_list:
        img = cv2.imread(image)
        img = cv2.resize(img, (width, height))
        video.write(img)
    video.release()

    """
    target_dirs = os.listdir(target_dir)
    target_dirs = [target for target in target_dirs if target.find('.DS_Store') == -1]
    for target in target_dirs:
        images = glob.glob(os.path.join(target_dir, target, '*.jpg'))
        images.sort()
        first_img = cv2.imread(images[0])
        height, width = first_img.shape[:2]

        video_file = os.path.join(target_dir, target, target + '.mp4')
        video = cv2.VideoWriter(video_file, fourcc, 5.0, (width, height))
        print(video_file)

        for image in images:
            img = cv2.imread(image)
            video.write(img)
        video.release()
    """

def main():
    anno_csv = os.path.join(root_dir, 'data', 'anno', 'battle.csv')
    train_anno_csv = os.path.join(root_dir, 'data', 'anno', 'train.csv')
    data_dir = os.path.join('/usr', 'local', 'wk', 'data', 'fortnite')
    image_dir = os.path.join(data_dir, 'image')
    movie_dir = os.path.join(data_dir, 'movie')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    # アノテーションから作成
    # make_from_anno(anno_csv, image_dir, movie_dir, fourcc)

    # ディレクトリから作成(アノテーションから戦闘 or Notで分けて作成)
    annotation_file = os.path.join(data_dir, 'annotation.list')
    make_from_dir(data_dir, fourcc, annotation_file)

if __name__ == '__main__':
    main()
