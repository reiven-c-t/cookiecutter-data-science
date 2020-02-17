from config.const import *
import platform


MODEL = "model-mix-qs"
TOKENIZER = "tokenizer-mix-qs"
CONFIG = "config-mix-qs"

DEVICE = "cuda"

TITLE = "title"
DESCRIPTION = "description"
QUALITY_SCORE = "quality_score"


MAX_SEQ_LEN = 40
MAX_INPUT_LEN = 100


ROUGH_INITIAL_TRAIN_EVAL = 3
# ROUGH_INITIAL_TRAIN_EVAL = None

if platform.system() == "Darwin":
    DEVICE = "cpu"
