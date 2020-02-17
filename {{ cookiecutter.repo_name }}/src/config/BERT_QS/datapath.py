from config.datapath import *
from os import path

MODEL_PATH = path.join(PROJECT_TOP, "models", "bert_qs")
TENSORBOARD_PATH = path.join(DATA_DIR, "tensorboard", "BERT_qs")
RESOURCE_PATH = path.join(DATA_DIR, "processed", "2019_10_04_system")
RESOURCE_TRAIN_PATH = path.join(RESOURCE_PATH, "train.tsv")
RESOURCE_DEV_PATH = path.join(RESOURCE_PATH, "dev.tsv")
RESOURCE_TEST_PATH = path.join(RESOURCE_PATH, "test.tsv")
