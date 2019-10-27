# coding=utf-8
import os
import subprocess


def parse_battle_csv(line):
    line = line.rstrip()
    vals = line.split(',')
    no = vals[0]
    name = vals[1]
    length = vals[2]
    url = vals[3]
    battle_time = vals[4]
    battle_timings = battle_time.split(' ')
    return no, name, length, url, battle_time, battle_timings


def change_from_hms_to_sec(hms_list):
    if len(hms_list) == 2:
        minute = int(hms_list[0])
        second = int(hms_list[1])
        total_sec = minute * 60 + second
    elif len(hms_list) == 3:
        hour = int(hms_list[0])
        minute = int(hms_list[1])
        second = int(hms_list[2])
        total_sec = hour * 60 * 60 + minute * 60 + second

    return total_sec


def extract_movie(in_movie_file, out_movie_file, start_sec, length_sec, high_quarity_flg=False):
    if high_quarity_flg:
        ffmpeg_cmd = 'ffmpeg -i ' + in_movie_file + ' -ss ' + str(start_sec) + ' -t ' + str(length_sec) + ' ' + out_movie_file
    else:
        ffmpeg_cmd = ' '.join(['ffmpeg', '-i', in_movie_file, '-ss', str(start_sec), '-t', str(length_sec),
                           '-vcodec', 'copy', '-acodec', 'copy', '-c', 'copy', out_movie_file])
    subprocess.call(ffmpeg_cmd.split())
    print('[info] in_movie_file:{}, crop_sec:{}~{}, out_movie_file:{}'.format(
        os.path.basename(in_movie_file), start_sec, start_sec + length_sec, out_movie_file))
