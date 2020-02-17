from time import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from config.BERT_QS.const import TOKENIZER, MODEL, DEVICE
from config.BERT_QS.datapath import RESOURCE_TEST_PATH
from models.BERT_QS.evaluate_model import evaluate_model
from utils.set_seed import set_seed
from utils.model_name_or_path import model_path_or_name
from utils.get_tokenizer import get_tokenizer
from data.BERT_QS.data_manager import DataManager
from models.BERT_QS.model import config
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(model_path_or_name(MODEL, suffix="es_eval_loss_best"), config=config)
set_seed(42)
# NOTE: es_eval_loss_best is minimum dev loss. Not eval(2019-10-08) loss.
DEVICE = "cuda"
BATCH_SIZE = 1024

tokenizer = get_tokenizer(TOKENIZER)

data_manager = DataManager(tokenizer, batch_size=BATCH_SIZE)
test_data_iterator = data_manager.load_file(RESOURCE_TEST_PATH, clear_cache=False)

losses, trues, preds = list(), list(), list()
with torch.no_grad():
    model.to(DEVICE)
    model.eval()
    for batch in test_data_iterator:
        inputs = data_manager.fix_batch(batch, device=DEVICE)
        outputs = model(**inputs)
        batch_size = outputs[1].shape[0]
        losses.append(outputs[0].item())
        trues.append(inputs["labels"].view(batch_size))
        preds.append(outputs[1].view(batch_size))
    losses = np.mean(losses)
    trues = torch.cat(trues).to("cpu").numpy()
    preds = torch.cat(preds).to("cpu").numpy()

    print("mse:", mean_squared_error(trues, preds))
    trues = trues.astype(np.int)
    preds = np.round(preds).astype(np.int)
    print("Acc:", accuracy_score(trues, preds))
    print("loss:", losses)

