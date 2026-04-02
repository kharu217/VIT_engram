import torch
import torch.nn.functional as F
from torch import Tensor, nn
from dataclasses import dataclass
from modules import MSA_Encoder, MOE_Encoder

@dataclass
class TETConfig:
    vocab_size=49408
    max_ctx_len: int = 77
    emb_dim: int = 128
    n_heads: int = 8
    attn_mask: Tensor = None
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
        self.embedding = nn.Embedding(num_embeddings=vocab_n, embedding_dim=embed_dim, padding_idx=77)
        self.pos_emb = nn.Parameter(torch.empty(max_ctx_n, embed_dim))
    
    def forward(self, x) :
        x = self.embedding(x) + self.pos_emb
        return x

class TET(nn.Module) :
    def __init__(self, cfg:TETConfig):
        super().__init__()
        self.cfg = cfg

        self.use_moe = cfg.use_moe
        
        self.token_emb = token_embedding(vocab_n=cfg.vocab_size, embed_dim=cfg.emb_dim, max_ctx_n=cfg.max_ctx_len)
        self.Encoder = MOE_Encoder(ffn_dropout=cfg.ffn_dropout,
                                   attn_dropout=cfg.attn_dropout,
                                   mask=cfg.attn_mask,
                                   depth=cfg.depth,
                                   emb_dim=cfg.emb_dim,
                                   ffn_mul=cfg.ffn_mul,
                                   n_heads=cfg.n_heads,
                                   c=cfg.c,
                                   k=cfg.k,
                                   n_experts=cfg.n_experts,
                                   every_2=cfg.every_2
                                   ) if cfg.use_moe else MSA_Encoder(ffn_dropout=cfg.ffn_dropout,
                                   mask=cfg.attn_mask,
                                   attn_dropout=cfg.attn_dropout,
                                   depth=cfg.depth,
                                   emb_dim=cfg.emb_dim,
                                   ffn_mul=cfg.ffn_mul,
                                   n_heads=cfg.n_heads)

    def forward(self, x) :
        emb = self.token_emb(x)
        if self.use_moe :
            out, aux_loss = self.Encoder(emb)
        else :
            out = self.Encoder(emb)
        
        if self.use_moe :
            return out, aux_loss
        return out

