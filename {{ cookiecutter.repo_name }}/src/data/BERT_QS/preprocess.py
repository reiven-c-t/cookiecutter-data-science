import pandas as pd
import codecs
import csv
from os import path as path


rawpath = "/Users/s09518/Documents/Python/QS_Estimator/data/raw/system_preprocess/2019-10-04-debug.tsv"
processed_path = "/Users/s09518/Documents/Python/QS_Estimator/data/processed/2019_10_04_system"
data = list()
with codecs.open(rawpath, "r", "utf-8") as f:
    for line in f.readlines():
        line_item = line.strip().split("\t")
        if len(line_item)!= 9:
            print(line_item)
        else:
            data.append(line.strip())
with codecs.open(rawpath, "w", "utf-8") as f:
    for line in data:
        f.write(line + "\n")

df = pd.read_csv(rawpath, delimiter="\t")

df = df.dropna()
print(df.shape)
print(df.columns)
df = df.rename(columns={"QS": "quality_score"})
df = df.reset_index(drop=True)
index_range = range(df.index.start, df.index.stop)

import numpy as np
np.random.seed(42)
train_index = np.random.choice(index_range, int(len(index_range) * 0.8), replace=False)
train_index.sort()
rest_index = list(set(index_range) - set(train_index))
dev_index = np.random.choice(rest_index, int(len(rest_index) * 0.5), replace=False)
dev_index.sort()
test_index = list(set(rest_index) - set(dev_index))
test_index.sort()

train_pd = df.iloc[train_index, :]
dev_pd = df.iloc[dev_index, :]
test_pd = df.iloc[test_index, :]
train_pd.to_csv(path.join(processed_path, "train.tsv"), sep="\t")
test_pd.to_csv(path.join(processed_path, "test.tsv"), sep="\t")
dev_pd.to_csv(path.join(processed_path, "dev.tsv"), sep="\t")

