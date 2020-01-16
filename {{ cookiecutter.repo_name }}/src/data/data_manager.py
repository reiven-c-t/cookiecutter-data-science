from torchtext.data import RawField
from torchtext.data import BucketIterator
from config.const import BATCH_SIZE, DO_LOWER_CASE, MAX_SEQ_LEN, FIELD_NAME, DEVICE
from transformers import BertTokenizer
from torchtext.data import Dataset, Example
import torch
from os import path
import codecs


class DataManager:
    def __init__(self, tokenizer: BertTokenizer, seq_max_len=MAX_SEQ_LEN, batch_size=BATCH_SIZE,
                 do_lower_case=DO_LOWER_CASE, field_name=FIELD_NAME, device=DEVICE):
        self.batch_size = batch_size
        self.max_len = seq_max_len
        self.text = RawField(preprocessing=lambda text: self.preprocess_text(text))
        self.field_name = field_name
        self.do_lower_case = do_lower_case
        self.device = device
        self.to_torch_long = True
        self.tokenizer = tokenizer

    def generate_example(self, line):
        return Example.fromlist([line], [(self.field_name, self.text)])

    def generate_examples_from_path(self, filepath):
        examples = list()
        with codecs.open(filepath, "r", "utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if self.do_lower_case:
                    line = line.lower()
                example = self.generate_example(line)
                examples.append(example)
        return examples

    def load_file(self, filepath, manage_cache=True, cache_ext=".cache", do_eval=False, clear_cache=False):
        if manage_cache:
            cache_filepath = filepath + cache_ext
            if path.isfile(cache_filepath) and not clear_cache:  # if cache file is exists, then load it
                dataset_examples = torch.load(cache_filepath)
                dataset = Dataset(dataset_examples, fields={self.field_name: self.text})
            else:  # otherwise, save cache
                dataset_examples = self.generate_examples_from_path(filepath)
                dataset = Dataset(dataset_examples, fields={self.field_name: self.text})
                torch.save(dataset.examples, cache_filepath)
        else:
            dataset_examples = self.generate_examples_from_path(filepath)
            dataset = Dataset(dataset_examples, fields={self.field_name: self.text})
        if do_eval:
            iterator = BucketIterator(dataset, batch_size=self.batch_size, shuffle=False)
            return iterator
        iterator = BucketIterator(dataset=dataset, batch_size=self.batch_size, shuffle=False) # TODO: 本質的にはTrue
        return iterator

    def pad_sequence(self, sequence):
        seq_len = len(sequence)
        if seq_len < self.max_len:
            pad = [self.tokenizer.pad_token_id] * (self.max_len - seq_len)
            sequence = sequence + pad
        if seq_len > self.max_len:
            sequence = sequence[0: self.max_len]
        return sequence

    def preprocess_text(self, sentence, add_cls=True, add_sep=True):
        sentence = sentence.strip()
        if self.do_lower_case:
            sentence = sentence.lower()
        input_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sentence))
        if add_cls:
            input_ids = [self.tokenizer.cls_token_id] + input_ids
        if add_sep:
            input_ids = input_ids + [self.tokenizer.sep_token_id]
        input_ids = self.pad_sequence(input_ids)
        return input_ids

    def fix_batch(self, text, device="cpu"):
        result = torch.tensor(text, dtype=torch.long, device=device)
        return result


if __name__ == '__main__':
    pass
