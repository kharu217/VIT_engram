import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from dataclasses import dataclass
from modules import MSA_Encoder, MOE_Encoder
from engram import engram_config

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
    device="cuda"


class PatchEmbedding(nn.Module):
    def __init__(self, cfg:ViTConfig, engram_cfg:engram_config=None):
        super().__init__()

        self.engram_cfg = engram_cfg

        self.patch_size = cfg.patch_size
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(cfg.in_channels, cfg.emb_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        if engram_config :
            self.to_dis_token = nn.Sequential(
                # using a conv layer instead of a linear one -> performance gains
                nn.Conv2d(cfg.in_channels, engram_config.vocab_size, kernel_size=cfg.patch_size, stride=cfg.patch_size),
                Rearrange('b e (h) (w) -> b (h w) e'),
            )
            self.temparature = nn.Parameter(torch.log(torch.tensor(1.0)))

        self.positions = nn.Parameter(torch.randn((cfg.img_size // cfg.patch_size) **2, cfg.emb_dim))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        out = self.projection(x)
        engram_token = torch.argmax(F.gumbel_softmax(self.to_dis_token(x), hard=True, tau=torch.exp(self.temparature)), dim=2)
        # add position embedding
        out += self.positions
        return out, engram_token


class VIT(nn.Module) :
    def __init__(self, cfg:ViTConfig, engram_cfg:engram_config=None):
        super().__init__()
        self.use_moe = cfg.use_moe
        
        self.patch_emb = PatchEmbedding(cfg, engram_cfg=engram_cfg)

        if cfg.use_moe :
            self.Encoder = MOE_Encoder(ffn_dropout=cfg.ffn_dropout,
                                   attn_dropout=cfg.attn_dropout,
                                   depth=cfg.depth,
                                   emb_dim=cfg.emb_dim,
                                   ffn_mul=cfg.ffn_mul,
                                   n_heads=cfg.n_heads,
                                   c=cfg.c,
                                   k=cfg.k,
                                   n_experts=cfg.n_experts,
                                   every_2=cfg.every_2,
                                   engram_cfg=engram_cfg
                                   )
        else :
            self.Encoder = MSA_Encoder(ffn_dropout=cfg.ffn_dropout,
                                   attn_dropout=cfg.attn_dropout,
                                   depth=cfg.depth,
                                   emb_dim=cfg.emb_dim,
                                   ffn_mul=cfg.ffn_mul,
                                   n_heads=cfg.n_heads,
                                   engram_cfg=engram_cfg)

    def forward(self, x, engram_embedding_table=None) :
        emb, engram_token = self.patch_emb(x)
        if self.use_moe :
            out, aux_loss = self.Encoder(emb, engram_embedding_table, engram_token)
        else :
            out = self.Encoder(emb, engram_embedding_table, engram_token)
        
        if self.use_moe :
            return out, aux_loss
        return out
