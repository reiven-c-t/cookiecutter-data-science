from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from config.datapath import TENSORBOARD_PATH, RESOURCE_PATH
from config.const import DO_LOWER_CASE
from config.const import NUM_EPOCH, DEVICE, MAX_GRAD_NORM, DEFAULT_LEARNING_RATE, ADAM_EPSILON, WARMUP_STEP
from config.const import CONFIG, TOKENIZER, MODEL, ES_EPOCH
from utils.model_path_or_name import model_path_or_name

from utils.calc_loss import calc_loss
from utils.create_tbx import create_tbx
from data.data_manager import DataManager
from models.model import Seq2Seq
from time import time
import torch

from utils.tokenizer import tokenizer
from config.datapath import RESOURCE_TRAIN_PATH, RESOURCE_DEV_PATH

data_manager = DataManager(tokenizer)
train_data_iterator = data_manager.load_file(RESOURCE_TRAIN_PATH, clear_cache=True)
dev_data_iterator = data_manager.load_file(RESOURCE_DEV_PATH, clear_cache=True)

vocab_size = tokenizer.vocab_size
model = Seq2Seq


def train_iteration(model: Seq2Seq, data_iterator, optimizer, tbx, current_iter, device=DEVICE, print_step=1):
    epoch_loss = 0
    len_iter = 0
    model.train()
    model.to(device)
    for batch in data_iterator:
        current_iter += 1

        inputs = data_manager.fix_batch(batch.text, device=device)

        loss, outputs = model(loss_out=True)

        loss.backward()
        clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()

        # logging
        if current_iter % print_step == 0:
            tbx.add_scalar("iter/train_loss", loss.item(), current_iter)
            print("current:", current_iter)
        len_iter += 1
    epoch_loss = epoch_loss / len_iter
    return model, current_iter, epoch_loss, optimizer


def train(model: SentenceAutoEncoder, data_iterator, device=DEVICE, epochs=NUM_EPOCH, tensorboard_path=TENSORBOARD_PATH,
          tokenizer=tokenizer):
    model.train()
    model.to(device)
    optimizer = Adam(model.parameters(), lr=DEFAULT_LEARNING_RATE, eps=ADAM_EPSILON)

    tbx = create_tbx(tensorboard_path)

    current_iter = 0
    t_total = epochs * len(data_iterator)
    print_step = int(t_total * 0.01)

    start = time()

    prev_best_eval_loss = float('inf')
    es_wait_num_epoch = 0

    for epoch in range(epochs):
        model, current_iter, epoch_loss, optimizer = train_iteration(model, data_iterator,
                                                                     optimizer=optimizer, tbx=tbx,
                                                                     current_iter=current_iter,
                                                                     print_step=print_step)
        tbx.add_scalar("epoch/train_loss", epoch_loss, epoch)
        eval_loss = calc_loss(model, dev_data_iterator)

        if prev_best_eval_loss > eval_loss:
            prev_best_eval_loss = eval_loss
            es_wait_num_epoch = 0
            suffix = "es_eval_loss_best"
            model.save_pretrained(model_path_or_name(MODEL, suffix=suffix))
            tokenizer.save_pretrained(model_path_or_name(TOKENIZER, suffix=suffix))
        else:
            es_wait_num_epoch += 1
        if es_wait_num_epoch > ES_EPOCH and ES_EPOCH != 0:
            print("Early stopping by ", epoch)
            break

        print("epoch % 3d: train loss: %.3f: dev loss: %.3f, duration: %d, current_state: %d / %d" % (
            epoch, epoch_loss, eval_loss, time() - start, current_iter, t_total))

    model.save_pretrained(model_path_or_name(MODEL))
    tokenizer.save_pretrained(model_path_or_name(TOKENIZER))

    tbx.close()


if __name__ == '__main__':
    model.to(DEVICE)
    train(model, train_data_iterator)
