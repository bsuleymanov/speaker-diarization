import torch


def list_to_tensor(list_of_tensors):
    return torch.stack(list_of_tensors, dim=0)