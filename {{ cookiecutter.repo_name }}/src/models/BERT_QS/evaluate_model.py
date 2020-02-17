import torch
from config.BERT_QS.const import DEVICE
from utils.calc_accuracy import calc_accuracy


@torch.no_grad()
def evaluate_model(model, data_iterator, data_manager, device=DEVICE, rough_out=None):
    epoch_loss = 0
    len_iter = 0
    model.eval()
    model.to(device)
    print("\n\n\n\n\n\n")
    print("eval")

    for batch in data_iterator:
        if type(rough_out) == type(0):
            rough_out -= 1
            if rough_out < 0:
                break
        inputs = data_manager.fix_batch(batch, device=device)
        data_manager.print_batch(inputs)

        loss, outputs = model(**inputs)
        print("pred:",outputs[0].item())

        epoch_loss += loss.item()

        len_iter += 1
    print("\n\n\n\n\n\n")

    epoch_loss = epoch_loss / len_iter
    return epoch_loss
