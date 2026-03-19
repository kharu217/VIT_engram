import torch.nn as nn
from model.modules import MOE_Encoder
import torch

temp = nn.Embedding(num_embeddings=10, embedding_dim=10, padding_idx=0)
temp2 = nn.Parameter(torch.empty((77, 10)))
test = torch.randint(0, 10, (10, 10))

y = temp(test)
y = y + temp2
print(y.shape)