def pad_tensor(tensor, pad_length, pad_item):
    """
    :param tensor: 1 * seq_len, seq_len <= pad_length
    :param pad_length:
    :param pad_item:
    :return:
    """
    tensor = tensor.to("cpu").clone()
    if pad_length - tensor.shape[1] > 0:
        pad = torch.ones(pad_length - tensor.shape[1]) * pad_item
        pad = pad.unsqueeze(0)
        return torch.cat((tensor, pad), dim=1)
    else:
        return tensor
