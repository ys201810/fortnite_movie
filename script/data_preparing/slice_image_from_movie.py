# coding=utf-8
import os
import cv2
import math
import click


def check_movie_info(movie_file):
    cap = cv2.VideoCapture(movie_file)
    video_frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_len_sec = video_frame_num / video_fps
    print('video_name:{} fps:{}, frame_num:{}, sec:{}'.format(movie_file, video_fps, video_frame_num, video_len_sec))
    return video_fps


def save_image(movie_file, out_image_dir, fps):
    cap = cv2.VideoCapture(movie_file)
    if not cap.isOpened():
        return

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    n = 0
    save_num_per_sec = 5
    save_frame = int(fps / save_num_per_sec)

    while True:
        ret, frame = cap.read()
        if ret:
            sec = math.floor(n / fps)
            if n % save_frame == 0:
                cv2.imwrite(os.path.join(out_image_dir, '{}_{}_{}.{}'.format(
                    os.path.basename(movie_file).split('.')[0], str(sec), str(n).zfill(digit), 'jpg')), frame)
            n += 1
        else:
            return

@click.command()
@click.option('--movie_name', '-m', default='')
def main(movie_name):
    data_base_dir = os.path.join('/Users', 'shirai1', 'work', 'data', 'fortnite')
    image_dir = os.path.join(data_base_dir, 'image')
    movie_dir = os.path.join(data_base_dir, 'movie')
    movie_file = os.path.join(movie_dir, movie_name)
    print(movie_file)
    fps = check_movie_info(movie_file)

    movie_name = movie_name.replace('.mp4', '')

    out_image_dir = os.path.join(image_dir, movie_name)
    if not os.path.exists(out_image_dir):
        os.makedirs(out_image_dir, exist_ok=True)
    save_image(movie_file, out_image_dir, fps)


if __name__ == '__main__':
    main()
