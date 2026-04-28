import torch
import torch.nn.functional as F
from torch import Tensor, nn
from dataclasses import dataclass
from modules import MSA_Encoder, MOE_Encoder
from engram import engram_config

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class TETConfig:
    vocab_size:int=49408
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
    device="cuda"
    engram_cfg:engram_config = None
    use_mhc:bool = False
    hc_mult: int = 4

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
        attn_mask = torch.triu(torch.full((cfg.max_ctx_len, cfg.max_ctx_len), float("-inf")), diagonal=1).to(device)

        self.use_moe = cfg.use_moe
        
        self.token_emb = token_embedding(vocab_n=cfg.vocab_size, embed_dim=cfg.emb_dim, max_ctx_n=cfg.max_ctx_len)
        if cfg.use_moe :
            self.Encoder = MOE_Encoder(ffn_dropout=cfg.ffn_dropout,
                                    attn_dropout=cfg.attn_dropout,
                                    mask=attn_mask,
                                    depth=cfg.depth,
                                    emb_dim=cfg.emb_dim,
                                    ffn_mul=cfg.ffn_mul,
                                    n_heads=cfg.n_heads,
                                    c=cfg.c,
                                    k=cfg.k,
                                    n_experts=cfg.n_experts,
                                    every_2=cfg.every_2,
                                    engram_cfg=cfg.engram_cfg,
                                    use_mhc=cfg.use_mhc,
                                    hc_mult=cfg.hc_mult
                                    ) 
        else :
            self.Encoder = MSA_Encoder(ffn_dropout=cfg.ffn_dropout,
                                    mask=attn_mask,
                                    attn_dropout=cfg.attn_dropout,
                                    depth=cfg.depth,
                                    emb_dim=cfg.emb_dim,
                                    ffn_mul=cfg.ffn_mul,
                                    n_heads=cfg.n_heads,
                                    use_mhc=cfg.use_mhc,
                                    hc_mult=cfg.hc_mult,
                                    engram_cfg=cfg.engram_cfg)

    def forward(self, x, engram_embedding_table=None) :
        emb = self.token_emb(x)
        if self.use_moe :
            out, aux_loss = self.Encoder(emb, engram_embedding_table ,x)
        else :
            out = self.Encoder(emb, engram_embedding_table ,x)
        
        if self.use_moe :
            return out, aux_loss
        return out

