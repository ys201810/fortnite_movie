# coding=utf-8
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


