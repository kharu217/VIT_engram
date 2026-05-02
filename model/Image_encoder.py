import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from dataclasses import dataclass
from modules import MSA_Encoder, MOE_Encoder
from engram import EngramConfig

@dataclass
class VitConfig:
    # Input
    in_channels = 3,
    img_size = 224,
    patch_size = 16,

    # core
    emb_dim = 128,
    depth = 16,

    # Attention
    n_heads = 8,
    attn_dropout = 0.1,

    # FFN
    ffn_mul = 4,
    ffn_dropout = 0.1,

    # MoE
    use_moe = False,
    n_experts = 16,
    k = 1,
    c = 1.0,
    every_2 = False,

    # MHC
    use_mhc = False,
    hc_mult = 4,
    engram_cfg = None,

    #etc
    device="cuda"

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, img_size, patch_size, emb_dim, engram_vocab_size):
        super().__init__()

        self.patch_size = patch_size
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        if engram_vocab_size :
            self.to_dis_token = nn.Sequential(
                # using a conv layer instead of a linear one -> performance gains
                nn.Conv2d(in_channels, engram_vocab_size, kernel_size=patch_size, stride=patch_size),
                Rearrange('b e (h) (w) -> b (h w) e'),
            )
            self.temparature = nn.Parameter(torch.log(torch.tensor(1.0)))

        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2, emb_dim))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        out = self.projection(x) + self.positions

        if self.to_dis_token :
            engram_token = torch.argmax(F.gumbel_softmax(self.to_dis_token(x), hard=True, tau=torch.exp(self.temparature)), dim=2)
            return out, engram_token
        else :
            return out


class VIT(nn.Module) :
    def __init__(self, cfg:VitConfig, engram_cfg:EngramConfig):
        super().__init__()
        self.use_moe = cfg.use_moe
        
        self.patch_emb = PatchEmbedding(cfg, engram_cfg)

        if cfg.use_moe :
            self.Encoder = MOE_Encoder(ffn_dropout=cfg.ffn_dropout,
                                   attn_dropout=cfg.attn_dropout,
                                   depth=cfg.depth,
                                   emb_dim=cfg.emb_dim,
                                   ffn_mul=cfg.ffn_mul,
                                   n_heads=cfg.n_heads,
                                   c=cfg.c,
                                   k=cfg.k,
                                   use_mhc=cfg.use_mhc,
                                   n_experts=cfg.n_experts,
                                   every_2=cfg.every_2,
                                   hc_mult=cfg.hc_mult,
                                   engram_cfg=engram_cfg
                                   )
        else :
            self.Encoder = MSA_Encoder(ffn_dropout=cfg.ffn_dropout,
                                   attn_dropout=cfg.attn_dropout,
                                   depth=cfg.depth,
                                   emb_dim=cfg.emb_dim,
                                   ffn_mul=cfg.ffn_mul,
                                   n_heads=cfg.n_heads,
                                   hc_mult=cfg.hc_mult,
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
