# coding=utf-8
import os
import sys
import glob
import math
import copy
import shutil
from script.utils import parse_battle_csv, change_from_hms_to_sec
root_dir = os.path.join(os.getcwd(), '../../')
sys.path.append(root_dir)


def main():
    data_dir = os.path.join('/usr', 'local', 'wk', 'data', 'fortnite')
    anno_csv = os.path.join(root_dir, 'data', 'anno', 'battle.csv')
    train_anno_csv = os.path.join(root_dir, 'data', 'anno', 'train.csv')
    image_dir = os.path.join(data_dir, 'image')
    movie_dir = os.path.join(data_dir, 'movie')
    train_image_dir = os.path.join(data_dir, 'train')
    val_image_dir = os.path.join(data_dir, 'val')

    with open(anno_csv, 'r') as inf:
        for i, line in enumerate(inf):
            if i == 0 or i == 1:
                continue

            no, name, length, url, battle_time, battle_timings = parse_battle_csv(line)
            kind = 'train' if no != 8 else 'val'
            # 1. 各動画の5sの最大値を取得
            max_sec = change_from_hms_to_sec(length.split(':'))
            max_5_sec = math.floor(max_sec / 5) * 5
            if max_sec == max_5_sec:
                max_5_sec = max_5_sec - 5
            print('target_num:{}, max_sec:{}, max_5_sec:{}'.format(no, max_sec, max_5_sec))

            # 2. battle_timeの開始終了リスト作成
            battle_secs = []
            for battle_timing in battle_timings:
                start_time, end_time = battle_timing.split('~')[0], battle_timing.split('~')[1]
                start_sec = change_from_hms_to_sec(start_time.split(':'))
                end_sec = change_from_hms_to_sec(end_time.split(':'))
                battle_secs.append([start_sec, end_sec])
            print(battle_secs)

            # 3. 5秒ごとに1秒1枚ずつ画像を取得(1秒1フレームを5枚合わせて利用。)
            target_image_dir = os.path.join(image_dir, no)
            sec_5_images = []
            images_anno_dict = {}
            for i in range(1, max_5_sec + 1):
                each_sec_images = glob.glob(os.path.join(target_image_dir, no + '_' + str(i) + '_*.jpg'))
                each_sec_first_image = each_sec_images[0]
                sec_5_images.append(each_sec_first_image)
                if i % 5 == 0:
                    sec_5_images_tmp = copy.copy(sec_5_images)
                    images_anno_dict[str(i - 4) + '~' + str(i)] = sec_5_images_tmp
                    sec_5_images.clear()
                if i == max_5_sec:
                    break

            # 4. 各5秒がbattle_timingに含まれるかどうかの判定
            anno_text = ''
            for key in images_anno_dict.keys():
                battle_flg = 0
                sec_start, sec_end = int(key.split('~')[0]), int(key.split('~')[1])
                for battle_sec in battle_secs:
                    if battle_sec[0] <= sec_start <= battle_sec[1] or battle_sec[0] <= sec_end <= battle_sec[1]:
                        battle_flg = 1
                        break

                # 5. フラグでtrainとvalに画像をコピー・アノテーションのテキストを作成。
                target_anno_dir = os.path.join(train_image_dir, no + '_' + key.replace('~', '_'))
                os.makedirs(target_anno_dir)
                for image in images_anno_dict[key]:
                    shutil.copy(image, os.path.join(target_anno_dir, os.path.basename(image)))
                anno_str = target_anno_dir + '/*.jpg ' + kind + ' ' + str(battle_flg) +'\n'
                anno_text = anno_text + anno_str
                print(anno_str)

            with open(os.path.join(data_dir, 'annotation_' + str(no) + '.list'), 'w') as outf:
                outf.write(anno_text)


if __name__ == '__main__':
    main()
