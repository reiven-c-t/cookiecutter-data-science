from time import time
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification

from config.BERT_QS.const import NUM_EPOCH, ES_EPOCH, MAX_GRAD_NORM, DEFAULT_LEARNING_RATE, WEIGHT_DECAY, \
    ADAM_EPSILON, WARMUP_STEP, EXAMPLE_PRINT_AMOUNT
from config.BERT_QS.const import TOKENIZER, MODEL, DEVICE
from config.BERT_QS.datapath import TENSORBOARD_PATH
from config.BERT_QS.datapath import RESOURCE_TRAIN_PATH, RESOURCE_DEV_PATH
from models.BERT_QS.evaluate_model import evaluate_model
from utils.create_tbx import create_tbx
from utils.set_seed import set_seed
from utils.model_name_or_path import model_path_or_name
from utils.get_tokenizer import get_tokenizer
from data.BERT_QS.data_manager import DataManager
from models.BERT_QS.model import model

set_seed(42)

tokenizer = get_tokenizer(TOKENIZER, force_default=True)

data_manager = DataManager(tokenizer)
train_data_iterator = data_manager.load_file(RESOURCE_TRAIN_PATH, clear_cache=True)
dev_data_iterator = data_manager.load_file(RESOURCE_DEV_PATH, clear_cache=True)
print("load data")

def train_iteration(model: BertForSequenceClassification,
                    data_iterator, optimizer, scheduler, tbx, current_iter,prev_best_eval_loss,
                    device=DEVICE,
                    max_grad_norm=MAX_GRAD_NORM,
                    print_step=1, example_print_amount=EXAMPLE_PRINT_AMOUNT):
    len_iter = 0
    model.train()


    iter_loss = list()
    preds, losses = list(),list()

    for batch in data_iterator:
        current_iter += 1
        model.zero_grad()
        model.to(device)

        inputs = data_manager.fix_batch(batch, device=device)
        data_manager.print_batch(inputs)
        outputs = model(**inputs)
        print(outputs[1][0, 0].item())

        loss = outputs[0]
        iter_loss.append(loss.item())
        loss.backward()

        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        optimizer.zero_grad()

        if example_print_amount > 0:
            data_manager.print_batch(inputs)
            example_print_amount -= 1

        # logging
        if current_iter % print_step == 0:
            tbx.add_scalar("iter/train_loss", np.mean(iter_loss), current_iter)
            iter_loss = list()
            learning_rate = scheduler.get_lr()[0]
            tbx.add_scalar("iter/lr", learning_rate, current_iter)
            eval_loss = evaluate_model(model, dev_data_iterator, data_manager)
            tbx.add_scalar("iter/eval_loss", eval_loss, current_iter)
            print("current:", current_iter)

            if prev_best_eval_loss > eval_loss:
                prev_best_eval_loss = eval_loss
                suffix = "es_eval_loss_best"
                model.save_pretrained(model_path_or_name(MODEL, suffix=suffix))
                tokenizer.save_pretrained(model_path_or_name(TOKENIZER, suffix=suffix))


        len_iter += 1
    epoch_loss = np.mean(losses)
    return model, current_iter, epoch_loss, optimizer, scheduler, prev_best_eval_loss


def train(model: BertForSequenceClassification,
          data_iterator,
          device=DEVICE,
          epochs=NUM_EPOCH,
          tensorboard_path=TENSORBOARD_PATH,
          tokenizer=tokenizer,
          leanring_rate=DEFAULT_LEARNING_RATE,
          adam_eps=ADAM_EPSILON,
          weight_decay=WEIGHT_DECAY,
          es_epoch=ES_EPOCH,
          warm_up_step=WARMUP_STEP):

    tbx = create_tbx(tensorboard_path)

    current_iter = 0
    t_total = epochs * len(data_iterator)
    print_step = int(t_total * 0.05)

    model.train()
    model.to(device)
    model.zero_grad()

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=leanring_rate, eps=adam_eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_step, num_training_steps=t_total)

    start = time()

    prev_best_eval_loss = float('inf')
    es_wait_num_epoch = 0

    epoch = 0
    epoch_loss = evaluate_model(model, train_data_iterator, data_manager)
    tbx.add_scalar("epoch/train_loss", epoch_loss, epoch)
    eval_loss = evaluate_model(model, dev_data_iterator, data_manager)
    tbx.add_scalar("epoch/eval_loss", eval_loss, epoch)

    for epoch in range(1, epochs + 1):
        model, current_iter, epoch_loss, optimizer, scheduler, prev_best_eval_loss = train_iteration(model, data_iterator,
                                                                                optimizer=optimizer,
                                                                                scheduler=scheduler,
                                                                                tbx=tbx,
                                                                                current_iter=current_iter,
                                                                                print_step=print_step,
                                                                                prev_best_eval_loss=prev_best_eval_loss)
        tbx.add_scalar("epoch/train_loss", epoch_loss, epoch)
        eval_loss = evaluate_model(model, dev_data_iterator, data_manager)
        tbx.add_scalar("epoch/eval_loss", eval_loss, epoch)


        if prev_best_eval_loss > eval_loss:
            prev_best_eval_loss = eval_loss
            es_wait_num_epoch = 0
            suffix = "es_eval_loss_best"
            model.save_pretrained(model_path_or_name(MODEL, suffix=suffix))
            tokenizer.save_pretrained(model_path_or_name(TOKENIZER, suffix=suffix))
        else:
            es_wait_num_epoch += 1
        if es_wait_num_epoch > es_epoch and es_epoch != 0:
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
