import torch
from config.const import DEVICE
from models.model import Seq2Seq
from models.train_model import data_manager


@torch.no_grad()
def calc_loss(model:Seq2Seq, data_iterator, device=DEVICE):
    epoch_loss = 0
    len_iter = 0
    model.eval()
    model.to(device)
    for batch in data_iterator:
        inputs = data_manager.fix_batch(batch.text, device=device)

        loss, outputs = model(inputs, loss_out=True)
        epoch_loss += loss.item()

        len_iter += 1
    epoch_loss = epoch_loss / len_iter
    return epoch_loss