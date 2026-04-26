import torch
import torch.nn as nn
import torchinfo

temp = nn.Embedding((200000 * 2) * 2, 256).to(dtype=torch.float16)
torchinfo.summary(temp, input_data=[torch.randint(0, 10, (10, 77))])