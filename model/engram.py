#add engram
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

@dataclass
class engram_config :
    embd_d: int = 512
    engram_vocab_size: int = 226
    max_ngram: int = 3
    engram_embd_d: int = 1280
    n_streams: int = 4
    engram_layer_n = [2, 8]
    vocab_size: int = 49408

class ShortConv(nn.Module):
    def __init__(self, hidden_size, kernel_size=4, dilation=1, hc_mult=4):
        super().__init__()
        self.hc_mult = hc_mult
        total_channels = hidden_size * hc_mult

        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,  # Depthwise
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.norms = nn.ModuleList([nn.RMSNorm(hidden_size) for _ in range(hc_mult)])
        self.act = nn.SiLU()

    def forward(self, x):
        B, T, G, C = x.shape
        normed_chunks = [self.norms[i](x[:, :, i, :]) for i in range(G)]
        x_norm = torch.cat(normed_chunks, dim=-1)

        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T]

        y = self.act(y_bct)
        return y.transpose(1, 2).view(B, T, G, C)


class NgramHashMapping(nn.Module):
    def __init__(self, vocab_sizes, max_ngram, num_heads=2, seed=42):
        super().__init__()
        self.vocab_sizes = [vocab_sizes] * (max_ngram - 1)
        self.max_ngram = max_ngram
        self.num_heads = num_heads

        self.register_buffer(
            'multipliers',
            torch.randint(1, 10000, (max_ngram - 1, num_heads, max_ngram))
        )
        self.register_buffer(
            'modulos',
            torch.tensor([v for v in self.vocab_sizes]).view(-1, 1)
        )

    def forward(self, input_ids):
        # input_ids are now BPE integers (can be large, e.g., 50256)
        # Hashing logic remains valid for any integer sequence
        padded = F.pad(input_ids, (self.max_ngram - 1, 0), value=0)
        windows = padded.unfold(dimension=1, size=self.max_ngram, step=1)
        all_hashes = []

        for n_idx in range(self.max_ngram - 1):
            ngram_len = n_idx + 2
            current_grams = windows[:, :, -ngram_len:]
            mults = self.multipliers[n_idx, :, :ngram_len]
            mixed = current_grams.unsqueeze(2) * mults.unsqueeze(0).unsqueeze(0)

            hashed = mixed[..., 0]
            for k in range(1, ngram_len):
                hashed = torch.bitwise_xor(hashed, mixed[..., k])

            hashed = hashed % self.modulos[n_idx]
            all_hashes.append(hashed)

        return torch.cat(all_hashes, dim=2)


class EngramModule(nn.Module):
    def __init__(self, engram_cfg:engram_config):
        super().__init__()

        self.engram_vocab_size = [engram_cfg.engram_vocab_size] * (engram_cfg.max_ngram - 1)
        self.embd_d = engram_cfg.embd_d
        self.engram_embd_d = engram_cfg.engram_embd_d
        self.n_streams = engram_cfg.n_streams

        self.hasher = NgramHashMapping(engram_cfg.engram_vocab_size, engram_cfg.max_ngram)
        self.total_slots = sum(self.hasher.vocab_sizes) * self.hasher.num_heads
        
        #self.embedding = nn.Embedding(total_slots, engram_cfg.engram_embd_d)

        total_ngrams = len(self.hasher.vocab_sizes) * self.hasher.num_heads
        input_dim = total_ngrams * engram_cfg.engram_embd_d

        self.val_proj = nn.Linear(input_dim, engram_cfg.embd_d)
        self.key_projs = nn.ModuleList([
            nn.Linear(input_dim, engram_cfg.embd_d) for _ in range(engram_cfg.n_streams)
        ])
        self.norm_q = nn.ModuleList([nn.RMSNorm(engram_cfg.embd_d) for _ in range(engram_cfg.n_streams)])
        self.norm_k = nn.ModuleList([nn.RMSNorm(engram_cfg.embd_d) for _ in range(engram_cfg.n_streams)])
        self.conv = ShortConv(engram_cfg.embd_d, hc_mult=engram_cfg.n_streams)

    def forward(self, x, input_ids, embedding:nn.Embedding):

        hash_ids = self.hasher(input_ids)
        offsets = torch.tensor([0] + [v for v in self.engram_vocab_size for _ in range(self.hasher.num_heads)],
                               device=x.device)
        offsets = torch.cumsum(offsets, dim=0)[:-1]
        hash_ids = hash_ids + offsets

        if embedding.weight.shape != torch.Size([self.total_slots, self.engram_embd_d]) :
            raise Exception("embedding table has unvalid shape")
        emb = embedding(hash_ids).flatten(start_dim=2)
        v_base = self.val_proj(emb)

        gates = []
        for i in range(self.n_streams):
            k = self.norm_k[i](self.key_projs[i](emb))
            q = self.norm_q[i](x[:, :, i, :])
            score = (q * k).sum(dim=-1, keepdim=True) / math.sqrt(self.embd_d)
            g = torch.sigmoid(score)
            gates.append(g)

        gates = torch.stack(gates, dim=2)
        v_gated = v_base.unsqueeze(2) * gates
        y = v_gated + self.conv(v_gated)
        return y

if __name__ == "__main__" :
    temp = NgramHashMapping([100, 100], 3)
    print(temp(torch.randint(0, 10, (5, 5))).shape)
    temp2 = EngramModule(128, [100, 100], max_ngram=3, engram_embd_d=128, n_streams=8)
    print(temp2(torch.randn((5, 10, 8, 128)), torch.randint(0, 10, (5, 10))).shape)

