# utils/denormalizer.py

import torch

def denormalize(img_tensor, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
    """
    img_tensor: C x H x W, torch.Tensor, 정규화된 상태 (mean/std 사용)
    반환: C x H x W, 0~1 범위 역정규화된 텐서
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    img_tensor = img_tensor * std + mean
    img_tensor = img_tensor.clamp(0, 1)
    return img_tensor