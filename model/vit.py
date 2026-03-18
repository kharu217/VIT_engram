import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchinfo import summary
from einops import repeat
from einops.layers.torch import Rearrange
from dataclasses import dataclass

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

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class SwitchGate(nn.Module):
    """
    SwitchGate module for MoE (Mixture of Experts) model.

    Args:
        dim (int): Input dimension.
        num_experts (int): Number of experts.
        capacity_factor (float, optional): Capacity factor for sparsity. Defaults to 1.0.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self,
        dim,
        num_experts: int,
        k: int,
        capacity_factor: float = 1.0,
        epsilon: float = 1e-6,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x: Tensor, use_aux_loss=False):
        """
        Forward pass of the SwitchGate module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Gate scores.
        """
        # Compute gate scores
        gate_scores = F.softmax(self.w_gate(x), dim=-1)

        # Determine the top-1 expert for each token
        capacity = int(self.capacity_factor * x.size(0))

        top_k_scores, top_k_indices = gate_scores.topk(k=int(self.k), dim=-1)

        # Mask to enforce sparsity
        mask = torch.zeros_like(gate_scores).scatter_(
            1, top_k_indices, 1
        )

        # Combine gating scores with the mask
        masked_gate_scores = gate_scores * mask

        # Denominators
        denominators = (
            masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        )

        # Norm gate scores to sum to the capacity
        gate_scores = (masked_gate_scores / denominators) * capacity

        if use_aux_loss:
            load = gate_scores.sum(0)  # Sum over all examples
            importance = gate_scores.sum(1).mean(0)  # Sum over all experts

            # Aux loss is mean suqared difference between load and importance
            loss = ((load - importance) ** 2).mean()

            return gate_scores, loss

        return gate_scores, None

class SwitchMoE(nn.Module): 
    """
    A module that implements the Switched Mixture of Experts (MoE) architecture.

    Args:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float, optional): The capacity factor that controls the capacity of the MoE. Defaults to 1.0.
        mult (int, optional): The multiplier for the hidden dimension of the feedforward network. Defaults to 4.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float): The capacity factor that controls the capacity of the MoE.
        mult (int): The multiplier for the hidden dimension of the feedforward network.
        experts (nn.ModuleList): The list of feedforward networks representing the experts.
        gate (SwitchGate): The switch gate module.

    """

    def __init__(
        self,
        dim: int,
        expansion: int,
        num_experts: int,
        k:int,
        capacity_factor: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.expansion = expansion
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor

        self.experts = nn.ModuleList(
            [
                FeedForwardBlock(dim, expansion)
                for _ in range(num_experts)
            ]
        )

        self.gate = SwitchGate(
            dim,
            num_experts,
            capacity_factor,
        )

    def forward(self, x: Tensor):
        """
        Forward pass of the SwitchMoE module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor of the MoE.

        """
        # (batch_size, seq_len, num_experts)
        gate_scores, loss = self.gate(
            x, use_aux_loss=True
        )

        # Dispatch to experts
        expert_outputs = [expert(x) for expert in self.experts]

        # Check if any gate scores are nan and handle
        if torch.isnan(gate_scores).any():
            print("NaN in gate scores")
            gate_scores[torch.isnan(gate_scores)] = 0

        # Stack and weight outputs
        stacked_expert_outputs = torch.stack(
            expert_outputs, dim=-1
        )  # (batch_size, seq_len, output_dim, num_experts)
        if torch.isnan(stacked_expert_outputs).any():
            stacked_expert_outputs[
                torch.isnan(stacked_expert_outputs)
            ] = 0

        # Combine expert outputs and gating scores
        moe_output = torch.sum(
            gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1
        )

        return moe_output, loss

class MSABLock(nn.Module) :
    def __init__(self, cfg:ViTConfig):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=cfg.emb_dim)
        self.MSA = nn.MultiheadAttention(embed_dim=cfg.emb_dim,
                                         num_heads=cfg.n_heads,
                                         dropout=cfg.attn_dropout,
                                         batch_first=True)
        self.FFN = FeedForwardBlock(emb_size=cfg.emb_dim,
                                    expansion=cfg.ffn_mul,
                                    drop_p=cfg.ffn_dropout)
    def forward(self, x) :
        out_1 = self.norm(x)
        attn_output, _ = self.MSA(out_1, out_1, out_1, need_weights=False)
        attn_output.add_(x)

        out_2 = self.norm(attn_output)
        out_2 = self.FFN(out_2) + out_1
        return out_2
    
class VMOEBLock(nn.Module) :
    def __init__(self, cfg:ViTConfig):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=cfg.emb_dim)
        self.MSA = nn.MultiheadAttention(embed_dim=cfg.emb_dim,
                                         num_heads=cfg.n_heads,
                                         dropout=cfg.attn_dropout,
                                         batch_first=True)
        self.moe = SwitchMoE(dim=cfg.emb_dim,
                             expansion=cfg.ffn_mul,
                             num_experts=cfg.n_experts,
                             k=cfg.k,
                             capacity_factor=cfg.c)
    def forward(self, x) :
        out_1 = self.norm(x)
        attn_output, _ = self.MSA(out_1, out_1, out_1, need_weights=False)
        attn_output.add_(x)

        out_2 = self.norm(attn_output)
        out_3, aux_loss = self.moe(out_2)
        
        out_3.add_(out_1)
        return out_3, aux_loss

class MSA_Encoder(nn.Sequential) :
    def __init__(self,
                 cfg:ViTConfig):
        super().__init__(*[MSABLock(cfg) for _ in range(cfg.depth)])

class VMOE_Encoder(nn.Module) :
    def __init__(self,
                 cfg:ViTConfig):
        super().__init__()
        self.layer = nn.ModuleList([layer for _ in range(cfg.depth//2) for layer in (MSABLock(cfg), VMOEBLock(cfg))])

    def forward(self, x) :
        aux_loss = 0
        for i, l in enumerate(self.layer) :
            if i%2==0 :
                x = l(x)
            else :
                x, loss = l(x)
                aux_loss += loss
        return x, aux_loss/len(self.layer)

class VIT(nn.Module) :
    def __init__(self, cfg:ViTConfig, class_n:int):
        super().__init__()
        self.use_moe = cfg.use_moe
        
        self.patch_emb = PatchEmbedding(cfg)
        self.Encoder = VMOE_Encoder(cfg) if cfg.use_moe else MSA_Encoder(cfg)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(cfg.emb_dim),
            nn.Linear(cfg.emb_dim, cfg.emb_dim),
            nn.GELU(),
            nn.Linear(cfg.emb_dim, class_n)
        )

    def forward(self, x) :
        emb = self.patch_emb(x)
        if self.use_moe :
            feat, aux_loss = self.Encoder(emb)
        else :
            feat = self.Encoder(emb)
        feat = feat.mean(dim=1).squeeze()
        out = self.cls_head(feat)
        
        if self.use_moe :
            return out, aux_loss
        return out

@dataclass
class vit_model :
    VIT_S_32 = ViTConfig(
            emb_dim=512,
            n_heads=8,
            ffn_mul=4,
            n_experts=32,
            depth=8,
            patch_size=32,
            c=1.05,
            k=1,
            use_moe=False
        )
    VMOE_S_32 = ViTConfig(
            emb_dim=512,
            n_heads=8,
            ffn_mul=4,
            n_experts=32,
            depth=8,
            patch_size=32,
            c=1.05,
            k=1,
            use_moe=True 
    )
 
if __name__ == "__main__" :
    test_model = VIT(vit_model.VIT_S_32, class_n=18291)
    test_model_moe = VIT(vit_model.VMOE_S_32, class_n=18291)

    import math
    print(round(summary(test_model, (10, 3, 224, 224), verbose=0).total_params/1000000, 1), "M")
    print(round(summary(test_model_moe, (10, 3, 224, 224), verbose=0).total_params/1000000, 1), "M")
