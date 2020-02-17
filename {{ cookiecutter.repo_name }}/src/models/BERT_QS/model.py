from config.BERT_QS.const import MODEL, CONFIG
from utils.model_name_or_path import model_path_or_name
from transformers import BertConfig
from transformers import BertForSequenceClassification

config = BertConfig.from_pretrained(model_path_or_name(CONFIG, force_default=True))
config.num_labels = 1

model = BertForSequenceClassification.from_pretrained(model_path_or_name(MODEL, force_default=True), config=config)
config.save_pretrained(model_path_or_name(CONFIG))
if __name__ == '__main__':
    pass
