import torch

print(torch.randn((10, 10)).shape)
print(torch.randn(10, 10).shape == torch.Size([10, 10]))

print(torch.nn.ModuleList([torch.nn.Embedding(10, 10), torch.nn.Embedding(10, 10)]).weight.shape)