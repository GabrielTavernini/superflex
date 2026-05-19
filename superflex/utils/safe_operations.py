import torch

def safe_pow(x, y, eps = 1e-3):
    safe_x = x.abs().clamp(min=eps, max =5e2)  # Prevents negatives & zero division
    return safe_x.pow(y).clamp(min=eps, max=5e2)

def safe_mul(x, y, eps = 1e-4):
    x = torch.sign(x) * torch.clamp(abs(x), eps, 1e6)
    y = torch.sign(y) * torch.clamp(abs(y), eps, 1e6)
    return x*y + eps