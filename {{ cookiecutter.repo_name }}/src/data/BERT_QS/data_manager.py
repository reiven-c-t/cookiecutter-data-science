from data.data_manager import DataManager as DataManagerMeta
from torchtext.data import RawField
from torchtext.data import BucketIterator
from config.BERT_QS.const import BATCH_SIZE, MAX_SEQ_LEN, MAX_INPUT_LEN, DEVICE
from config.BERT_QS.const import TITLE, DESCRIPTION, QUALITY_SCORE
from transformers import BertTokenizer
from torchtext.data import Dataset, Example
import torch
from os import path
import codecs
import pandas as pd


class DataManager(DataManagerMeta):
    def __init__(self, tokenizer: BertTokenizer, max_seq_len=MAX_SEQ_LEN, max_input_len=MAX_INPUT_LEN, batch_size=BATCH_SIZE,
                 title=TITLE, description=DESCRIPTION, quality_score=QUALITY_SCORE, device=DEVICE):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.max_input_len = max_input_len
        self.title = RawField(preprocessing=lambda text: self.preprocess_text(text))
        self.description = RawField(preprocessing=lambda text: self.preprocess_text(text))
        self.quality_score = RawField(preprocessing=lambda score: self.preprocess_level(score))
        self.title_label = title
        self.description_label = description
        self.quality_score_label = quality_score
        self.device = device
        self.to_torch_long = True
        self.tokenizer = tokenizer

    def generate_example(self, title, description, quality_score):
        return Example.fromlist([title, description, quality_score],
                                [(self.title_label, self.title),
                                 (self.description_label, self.description),
                                 (self.quality_score_label, self.quality_score)
                                 ])

    def generate_examples_from_path(self, tsv_data_filepath):
        examples = list()
        df = pd.read_csv(tsv_data_filepath, delimiter="\t")
        for index, row in df.iterrows():
            example = self.generate_example(row[self.title_label], row[self.description_label], row[self.quality_score_label])
            examples.append(example)
        return examples

    def load_file(self, filepath, manage_cache=True, cache_ext=".cache", do_eval=False, clear_cache=False):
        fields_dict = {self.title_label: self.title, self.description_label: self.description, self.quality_score_label: self.quality_score}
        if manage_cache:
            cache_filepath = filepath + cache_ext
            if path.isfile(cache_filepath) and not clear_cache:  # if cache file is exists, then load it
                dataset_examples = torch.load(cache_filepath)
                dataset = Dataset(dataset_examples,fields=fields_dict)
            else:  # otherwise, save cache
                dataset_examples = self.generate_examples_from_path(filepath)
                dataset = Dataset(dataset_examples,fields=fields_dict)
                torch.save(dataset.examples, cache_filepath)
        else:
            dataset_examples = self.generate_examples_from_path(filepath)
            dataset = Dataset(dataset_examples,fields=fields_dict)
        if do_eval:
            iterator = BucketIterator(dataset, batch_size=self.batch_size, shuffle=False)
            return iterator
        iterator = BucketIterator(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        return iterator

    def pad_sequence(self, sequence, pad_token_id=None, max_len=None):
        seq_len = len(sequence)
        if pad_token_id is not None:
            pad_token = [pad_token_id]
        else:
            pad_token = [self.tokenizer.pad_token_id]
        if max_len is None:
            max_len = self.max_seq_len

        if seq_len < max_len:
            pad = pad_token * (max_len - seq_len)
            sequence = sequence + pad
        if seq_len >= max_len:
            sequence = sequence[0: max_len]
        return sequence

    def preprocess_text(self, sentence, add_cls=False, add_sep=False):
        sentence = sentence.strip()
        input_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sentence))
        max_token_len = self.max_seq_len - int(add_cls) - int(add_sep)
        if len(input_ids) > max_token_len:
            input_ids = input_ids[0: self.max_seq_len - 2]
        if add_cls:
            input_ids = [self.tokenizer.cls_token_id] + input_ids
        if add_sep:
            input_ids = input_ids + [self.tokenizer.sep_token_id]
        # input_ids = self.pad_sequence(input_ids)
        return input_ids

    def preprocess_level(self, level):
        return float(level)

    def print_batch(self, batch):
        print()
        print()
        for key, value in batch.items():
            if key in ["input_ids", "attention_mask", "token_type_ids"]:
                print(key, value[0, :].tolist())
            elif key.find("label") != -1:
                print(key, value[0].item())

    def fix_batch(self, batch, device="cpu"):
        # inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": 0, "labels": batch[3]}
        inputs = dict()
        batch = batch.__dict__
        title = batch[self.title_label]
        description = batch[self.description_label]

        input_ids = []
        attention_masks = []
        token_type_ids = []
        for title_id_list, description_id_list in zip(title, description):
            attention_mask = [1]
            input_id = [self.tokenizer.cls_token_id]
            token_type_id = [0]
            for title_id in title_id_list:
                input_id.append(title_id)
                attention_mask.append(1)
                token_type_id.append(0)
            input_id.append(self.tokenizer.sep_token_id)
            attention_mask.append(1)
            token_type_id.append(0)
            for description_id in description_id_list:
                input_id.append(description_id)
                attention_mask.append(1)
                token_type_id.append(1)
            input_id.append(self.tokenizer.sep_token_id)
            attention_mask.append(1)
            token_type_id.append(1)

            input_id = self.pad_sequence(input_id, max_len=self.max_input_len)
            token_type_id = self.pad_sequence(token_type_id, 0, max_len=self.max_input_len)
            attention_mask = self.pad_sequence(attention_mask, 0, max_len=self.max_input_len)

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            token_type_ids.append(token_type_id)

        inputs["input_ids"] = torch.tensor(input_ids, dtype=torch.long, device=device)
        inputs["attention_mask"] = torch.tensor(attention_masks, dtype=torch.long, device=device)
        inputs["token_type_ids"] = torch.tensor(token_type_ids, dtype=torch.long, device=device)
        inputs["labels"] = torch.tensor(batch[self.quality_score_label], dtype=torch.float, device=device)

        return inputs


if __name__ == '__main__':
    from utils.get_tokenizer import get_tokenizer
    from config.BERT_QS.datapath import RESOURCE_TRAIN_PATH, RESOURCE_TEST_PATH, RESOURCE_DEV_PATH
    from os import path
    from config.BERT_QS.const import TOKENIZER

    tokenizer = get_tokenizer(tokenizer_label=TOKENIZER, force_default=True)
    data_manager = DataManager(tokenizer)
    # # data_loader = data_manager.load_file(RESOURCE_TRAIN_PATH, clear_cache=False)
    # # data_loader = data_manager.load_file(RESOURCE_DEV_PATH, clear_cache=False)
    # data_loader = data_manager.load_file(RESOURCE_TEST_PATH, clear_cache=False)
    # print(len(data_loader))
    # for batch in data_loader:
    #     pass

        # print(dir(batch))
        # print(data_manager.fix_batch(batch.text).shape)
        # break
