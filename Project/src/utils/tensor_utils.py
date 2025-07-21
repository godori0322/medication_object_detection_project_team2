
import torch


def collate_fn(batch):
    return tuple(zip(*batch))

def test_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images)
    return images, targets

def safe_tensor(data, shape, dtype):
    return torch.as_tensor(data, dtype=dtype) if len(data) else torch.zeros(shape, dtype=dtype)
