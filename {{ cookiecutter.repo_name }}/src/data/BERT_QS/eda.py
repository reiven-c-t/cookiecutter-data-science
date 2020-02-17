from utils.get_tokenizer import get_tokenizer
from config.BERT_QS.const import TOKENIZER
import numpy as np
import codecs

tokenizer = get_tokenizer(TOKENIZER)

seq1_len = list()
seq2_len = list()


def reader(filepath):
    result = list()
    with codecs.open(filepath, "r", "utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            tokenized = tokenizer.tokenize(line)
            result.append(len(tokenized))
    return result


seq_len = reader("/Users/reiven/Documents/Python/RewardExperiment/data/processed/Newsela/train.sentence.txt")
print(np.mean(seq_len))
print(np.mean(seq_len) + 1.96 * np.std(seq_len))

seq_len = reader("/Users/reiven/Documents/Python/RewardExperiment/data/processed/Newsela/dev.sentence.txt")
print(np.mean(seq_len))
print(np.mean(seq_len) + 1.96 * np.std(seq_len))

seq_len = reader("/Users/reiven/Documents/Python/RewardExperiment/data/processed/Newsela/2019-10-08.sentence.txt")
print(np.mean(seq_len))
print(np.mean(seq_len) + 1.96 * np.std(seq_len))
