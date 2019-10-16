import subprocess
from pathlib import Path


def save_src_files(src_dir, out_dir):
    """
    src_dirの中身を、out_dirにまるごと保存する
    例えば
    ./src/a.py
    ./src/b.py
    となっていたら、
    save_src_files('./src', './out')
    で、
    ./out/a.py
    ./out/b.py
    としてくれる
    """
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = src_dir.resolve()
    out_dir = out_dir.resolve()
    if str(src_dir) in str(out_dir):
        raise RuntimeError('Omg, {} is in {} !'.format(out_dir, src_dir))

    subprocess.call('cp -r {} {}'.format(src_dir, out_dir), shell=True)


