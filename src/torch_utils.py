import torch

def unbatch(tensor : torch.Tensor) -> torch.Tensor:
    """
    Takes an [n, k, N] tensor and reshapes into [n * k, N]
    """
    shape = tensor.shape
    return torch.reshape(tensor, [shape[0] * shape[1], shape[2]])

def custom_sigmoid(tensor : torch.Tensor) -> torch.Tensor:
    """
    Centered at 0.5, approx (0,0) to approx (1,1)
    """
    return 1 / (1 + torch.exp(-10 * tensor + 5))

def softplus(tensor : torch.Tensor) -> torch.Tensor:
    """
    \frac{\ln\left(1+e^{\left(\frac{1}{0.84}x-0.2\right)k}\right)}{k}
    """
    K = 30
    return torch.log(1 + torch.exp(1/84 * tensor - 0.2) ** K) / K