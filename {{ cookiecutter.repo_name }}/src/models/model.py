import torch
from torch import nn
from config.const import EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT_RATIO, DEVICE, MAX_SEQ_LEN, MODEL
from torch.nn import CrossEntropyLoss
from utils.model_path_or_name import model_path_or_name
from os import makedirs, path
from copy import deepcopy

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.saved_param = ["vocab_size", "emb_dim", "hid_dim", "num_layers", "dropout_ratio"]

    def forward(self, inputs):
        pass

    def save_pretrained(self, savepath):
        model_to_save = deepcopy(self)
        model_to_save = model_to_save.to("cpu")

        makedirs(savepath, exist_ok=True)
        torch.save(model_to_save.state_dict(), path.join(savepath, "state_dict.bin"))

        params = {}
        for key in self.__dict__.keys():
            if key in self.saved_param:
                params[key] = self.__dict__[key]
        torch.save(params, path.join(savepath, "params.bin"))

    @staticmethod
    def load_pretrained(loadpath):
        params = torch.load(path.join(loadpath, "params.bin"))
        model = Encoder(**params)
        model.load_state_dict(torch.load(path.join(loadpath, "state_dict.bin")))
        return model


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.saved_param = ["vocab_size", "emb_dim", "hid_dim", "num_layers", "dropout_ratio"]

    def forward(self):
        pass

    def save_pretrained(self, savepath):
        model_to_save = deepcopy(self)
        model_to_save = model_to_save.to("cpu")

        makedirs(savepath, exist_ok=True)
        torch.save(model_to_save.state_dict(), path.join(savepath, "state_dict.bin"))

        params = {}
        for key in self.__dict__.keys():
            if key in self.saved_param:
                params[key] = self.__dict__[key]
        torch.save(params, path.join(savepath, "params.bin"))

    @staticmethod
    def load_pretrained(loadpath):
        params = torch.load(path.join(loadpath, "params.bin"))
        model = Decoder(**params)
        model.load_state_dict(torch.load(path.join(loadpath, "state_dict.bin")))
        return model


class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.saved_params = ["vocab_size", "bos_token_id", "pad_token_id", "emb_dim", "num_layers", "hid_dim",
                             "dropout_ratio", "device", "max_seq_len"]

        if pad_token_id is not None:
            loss_func = CrossEntropyLoss(ignore_index=pad_token_id)
        else:
            loss_func = CrossEntropyLoss()
        self.loss_func = loss_func

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, inputs, loss_out=True):
        pass

    def encode(self, input_ids):
        hidden, cell = self.encoder()
        return hidden[-1]  # (num_layers * num_directions, batch, hidden_size)=(1, bs, hid_dim)

    def decode(self, input_ids, device="cpu"):
        hidden, cell = self.encoder()
        decoder_input_ids = torch.ones((input_ids.shape[0], 1), dtype=torch.long,
                                            device=device) * self.bos_token_id  # (batch_size, 1)

        len_count = 0
        while len_count < self.max_seq_len:
            dec_outputs, hidden, cell = self.decoder()
            next_token = dec_outputs.argmax(dim=2).permute(1,0)[:, -1].unsqueeze(1).to("cpu")
            decoder_input_ids = torch.cat((decoder_input_ids, next_token), dim=1)
            len_count += 1
        return decoder_input_ids

    def init_weight(self):
        for p in self.parameters():
            nn.init.xavier_uniform_(p)

    def save_pretrained(self, savepath):
        makedirs(savepath, exist_ok=True)
        model_to_save = deepcopy(self)
        model_to_save = model_to_save.to("cpu")
        encoder = model_to_save.encoder
        decoder = model_to_save.decoder

        enc_dir = path.join(savepath, "encoder")
        makedirs(enc_dir, exist_ok=True)
        encoder.save_pretrained(enc_dir)

        dec_dir = path.join(savepath, "decoder")
        makedirs(dec_dir, exist_ok=True)
        decoder.save_pretrained(dec_dir)

        params = {}
        for key in self.__dict__.keys():
            if key in self.saved_params:
                params[key] = self.__dict__[key]
        torch.save(params, path.join(savepath, "params.bin"))

    @staticmethod
    def from_pretrained(model_path):
        params = torch.load(path.join(model_path, "params.bin"))
        model = Seq2Seq(**params)

        enc_dir = path.join(model_path, "encoder")
        encoder = Encoder.load_pretrained(enc_dir)

        dec_dir = path.join(model_path, "decoder")
        decoder = Decoder.load_pretrained(dec_dir)

        model.encoder = encoder
        model.decoder = decoder

        return model


if __name__ == '__main__':
    pass

    # SAE.save_pretrained(model_path_or_name(MODEL))
