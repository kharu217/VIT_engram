import torch
import torch.nn.functional as F
from torch import Tensor, nn

class Fast_GELU(nn.Module) :
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

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
        drop_p: float = 0.1,
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
                FeedForwardBlock(dim, expansion, drop_p)
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


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            Fast_GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class MSABLock(nn.Module) :
    def __init__(self, emb_dim, n_heads,attn_dropout, ffn_mul, ffn_dropout, mask=None):
        super().__init__()
        self.mask = mask

        self.norm1 = nn.LayerNorm(normalized_shape=emb_dim)
        self.norm2 = nn.LayerNorm(normalized_shape=emb_dim)

        self.MSA = nn.MultiheadAttention(embed_dim=emb_dim,
                                         num_heads=n_heads,
                                         dropout=attn_dropout,
                                         batch_first=True)
        self.FFN = FeedForwardBlock(emb_size=emb_dim,
                                    expansion=ffn_mul,
                                    drop_p=ffn_dropout)
    def forward(self, x) :
        norm_x = self.norm1(x)
        feat, _ = self.MSA(norm_x, norm_x, norm_x, need_weights=False)
        x = x + feat
        out = self.FFN(self.norm2(x))
        out = out + x
        return out
    
class MOEBlock(nn.Module) :
    def __init__(self, emb_dim, n_heads,attn_dropout, ffn_mul, ffn_dropout, n_experts, k, c, mask=None):
        super().__init__()
        self.mask = mask
        self.norm1 = nn.LayerNorm(normalized_shape=emb_dim)
        self.norm2 = nn.LayerNorm(normalized_shape=emb_dim)
        self.MSA = nn.MultiheadAttention(embed_dim=emb_dim,
                                         num_heads=n_heads,
                                         dropout=attn_dropout,
                                         batch_first=True)
        self.moe = SwitchMoE(dim=emb_dim,
                             expansion=ffn_mul,
                             num_experts=n_experts,
                             k=k,
                             capacity_factor=c,
                             drop_p=ffn_dropout)
    def forward(self, x) :
        norm_x = self.norm1(x)
        feat, _ = self.MSA(norm_x, norm_x, norm_x)
        x = x + feat
        out, aux_loss = self.moe(self.norm2(x))
        out = out + x
        return out, aux_loss

class MSA_Encoder(nn.Sequential) :
    def __init__(self,emb_dim, n_heads, attn_dropout, ffn_mul, ffn_dropout, depth, mask=None):
        super().__init__(*[MSABLock(emb_dim=emb_dim,
                                    n_heads=n_heads,
                                    attn_dropout=attn_dropout,
                                    ffn_mul=ffn_mul,
                                    ffn_dropout=ffn_dropout,
                                    mask=mask) for _ in range(depth)])

class MOE_Encoder(nn.Module) :
    def __init__(self,emb_dim, n_heads, attn_dropout, ffn_mul, ffn_dropout, c, k, n_experts,depth, every_2, mask=None):
        super().__init__()
        if every_2 :
            self.layer = nn.ModuleList([layer for _ in range(depth//2) for layer in (
                            MSABLock(emb_dim=emb_dim,
                                        n_heads=n_heads,
                                        attn_dropout=attn_dropout,
                                        ffn_mul=ffn_mul,
                                        ffn_dropout=ffn_dropout, 
                                        mask=mask),

                            MOEBlock(emb_dim=emb_dim,
                                        n_heads=n_heads,
                                        attn_dropout=attn_dropout,
                                        ffn_mul=ffn_mul,
                                        ffn_dropout=ffn_dropout,
                                        c=c,
                                        k=k,
                                        n_experts=n_experts,
                                        mask=mask))])
        else :
            self.layer = nn.ModuleList([MSABLock(emb_dim=emb_dim,
                                    n_heads=n_heads,
                                    attn_dropout=attn_dropout,
                                    ffn_mul=ffn_mul,
                                    ffn_dropout=ffn_dropout, mask=mask) for _ in range(depth-2)])
            for _ in range(2) :
                self.layer.append(MOEBlock(emb_dim=emb_dim,
                                        n_heads=n_heads,
                                        attn_dropout=attn_dropout,
                                        ffn_mul=ffn_mul,
                                        ffn_dropout=ffn_dropout,
                                        c=c,
                                        k=k,
                                        n_experts=n_experts,
                                        mask=mask))


    def forward(self, x) :
        aux_loss = 0
        for l in self.layer :
            if l._get_name() != "MOEBlock" :
                x = l(x)
            else :
                x, loss = l(x)
                aux_loss += loss
        return x, aux_loss

if __name__ == "__main__" :
    import torchinfo
    test_model = MOE_Encoder(100, 4, 0.1, 1, 0.1, 1, 1, 16, 4, False)
    torchinfo.summary(test_model, input_size=(10, 32, 100)) 