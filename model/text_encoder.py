import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchinfo import summary
from einops.layers.torch import Rearrange
from dataclasses import dataclass
from modules import MSA_Encoder, MOE_Encoder

@dataclass
class ViTConfig:
    max_ctx_len: int = 77
    emb_dim: int = 128
    n_heads: int = 8
    attn_dropout: float = 0.1
    ffn_mul: int = 4
    ffn_dropout: float = 0.1
    n_experts: int = 16
    k: int = 1
    c: float = 1.0
    depth: int = 16
    use_moe:bool=False
    every_2:bool=False

class token_embedding(nn.Module) :
    def __init__(self, embed_dim:int, vocab_n:int, max_ctx_n:int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_n, embedding_dim=embed_dim)
        self.pos_emb = nn.Parameter(torch.empty(max_ctx_n, embed_dim))
    
    def forward(self, x) :
        x = self.embedding(x) + self.

class TTransformer(nn.Module) :

