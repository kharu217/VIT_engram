import torch
from torch import Tensor, nn
from einops.layers.torch import Rearrange
from dataclasses import dataclass
from modules import MSA_Encoder, MOE_Encoder

@dataclass
class ViTConfig:
    in_channels: int = 3
    patch_size: int = 16
    img_size: int = 224
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


class PatchEmbedding(nn.Module):
    def __init__(self, cfg:ViTConfig):
        self.patch_size = cfg.patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(cfg.in_channels, cfg.emb_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.positions = nn.Parameter(torch.randn((cfg.img_size // cfg.patch_size) **2, cfg.emb_dim))

        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        # add position embedding
        x += self.positions
        return x


class VIT(nn.Module) :
    def __init__(self, cfg:ViTConfig):
        super().__init__()
        self.use_moe = cfg.use_moe
        
        self.patch_emb = PatchEmbedding(cfg)
        self.Encoder = MOE_Encoder(ffn_dropout=cfg.ffn_dropout,
                                   attn_dropout=cfg.attn_dropout,
                                   depth=cfg.depth,
                                   emb_dim=cfg.emb_dim,
                                   ffn_mul=cfg.ffn_mul,
                                   n_heads=cfg.n_heads,
                                   c=cfg.c,
                                   k=cfg.k,
                                   n_experts=cfg.n_experts,
                                   every_2=cfg.every_2
                                   ) if cfg.use_moe else MSA_Encoder(ffn_dropout=cfg.ffn_dropout,
                                   attn_dropout=cfg.attn_dropout,
                                   depth=cfg.depth,
                                   emb_dim=cfg.emb_dim,
                                   ffn_mul=cfg.ffn_mul,
                                   n_heads=cfg.n_heads)

    def forward(self, x) :
        emb = self.patch_emb(x)
        if self.use_moe :
            out, aux_loss = self.Encoder(emb)
        else :
            out = self.Encoder(emb)
        
        if self.use_moe :
            return out, aux_loss
        return out
