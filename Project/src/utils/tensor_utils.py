
import torch


def collate_fn(batch):
    return tuple(zip(*batch))


def safe_tensor(data, shape, dtype):
    return torch.as_tensor(data, dtype=dtype) if len(data) else torch.zeros(shape, dtype=dtype)
