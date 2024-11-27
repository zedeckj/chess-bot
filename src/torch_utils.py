import torch

def unbatch(tensor : torch.Tensor) -> torch.Tensor:
    """
    Takes an [n, k, N] tensor and reshapes into [n * k, N]
    """
    shape = tensor.shape
    return torch.reshape(tensor, [shape[0] * shape[1], shape[2]])