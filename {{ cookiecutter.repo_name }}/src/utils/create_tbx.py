from os import makedirs, walk
from shutil import move
from tensorboardX import SummaryWriter

from config.datapath import TENSORBOARD_PATH


def create_tbx(tbx_path=TENSORBOARD_PATH):
    makedirs(tbx_path, exist_ok=True)
    makedirs(path.join(tbx_path, "old"), exist_ok=True)

    for current_dir, dirs, files in walk(tbx_path):
        for file in files:
            if file.find("events.out") == 0 and current_dir == tbx_path:
                move(path.join(tbx_path, file), path.join(tbx_path, "old"))
    tbx = SummaryWriter(logdir=tbx_path)
    return tbx
