"""
src以下で使用する全ての定数宣言

Default config setting.
Sub config for specific architecture should import this config and overwrite if necessary.

優先順位
config.const < config.domain.const < args
"""
from config.ArgsManager import args

TRAIN = "train"
DEV = "dev"
TEST = "test"
DATA_TYPE = [TRAIN, DEV, TEST]

MODEL_TYPE = "bert-base-japanese-whole-word-masking"

MAX_SEQ_LEN = 100
BATCH_SIZE = args.int(32, "--batch_size")
NUM_EPOCH = args.int(100, "--num_epoch")
WEIGHT_DECAY = 0.0

DEVICE = args.str(default="cuda", key="--device")

MAX_GRAD_NORM = 1.0
DEFAULT_LEARNING_RATE = 2e-5
ADAM_EPSILON = 1e-8
WARMUP_STEP = args.int(0, key='--warmup_step')
ES_EPOCH = args.int(3, key='--es_epoch')

EXAMPLE_PRINT_AMOUNT = 100000000

import platform
if platform.system() == "Darwin":
    DEVICE = "cpu"

post_fix = lambda x: x if platform.system() == "Darwin" else "cpu"