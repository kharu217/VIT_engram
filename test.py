import torch
from torch import tensor
import time

def sinkhorn_knopp(A: torch.Tensor, n_iter: int = 5) -> torch.Tensor:
    log_A = torch.log(A.abs() + 1e-8)
    for _ in range(n_iter):
        log_A = log_A - torch.logsumexp(log_A, dim=-1, keepdim=True)
        log_A = log_A - torch.logsumexp(log_A, dim=-2, keepdim=True)
    return torch.exp(log_A)

now = time.time()
temp = torch.randn((10, 10))

print(time.time() - now)