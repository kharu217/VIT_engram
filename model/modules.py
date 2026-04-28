import torch
import torch.nn.functional as F
from torch import Tensor, nn
from engram import EngramModule, engram_config

device = "cuda" if torch.cuda.is_available() else "cpu"

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

class mHyperConnection(nn.Module):
    def __init__(self, hidden_size: int, n: int, sinkhorn_iter: int = 20):
        super().__init__()
        self.n = n
        self.C = hidden_size
        self.sinkhorn_iter = sinkhorn_iter
        nC = n * hidden_size

        # 논문 eq(10~13): 파라미터를 하나로 통합
        # φ_l: [nC, n²+2n] — pre/post/res projection 통합
        self.phi = nn.Linear(nC, n * n + 2 * n, bias=False)  # φ_l

        # b_l: [1, n²+2n] — static bias 통합
        # 초기화: b_pre=0(sigmoid→0.5), b_post=0(2sigmoid→1.0), b_res=identity flatten
        b_init = torch.zeros(n * n + 2 * n)
        # b_res 부분을 identity로 초기화
        b_res_init = torch.eye(n).flatten()
        b_init[2 * n:] = b_res_init
        self.b = nn.Parameter(b_init)  # [n²+2n]

        # α scalars — small value 초기화
        self.alpha_pre  = nn.Parameter(torch.tensor(0.0))
        self.alpha_post = nn.Parameter(torch.tensor(0.0))
        self.alpha_res  = nn.Parameter(torch.tensor(0.0))

        # RMSNorm: nC 차원에 적용
        self.norm = nn.RMSNorm(nC)

    @staticmethod
    @torch.jit.script
    def sinkhorn_knopp(H_tilde: torch.Tensor, n_iter: int = 20) -> torch.Tensor:
        M = torch.exp(H_tilde)  # [B*L, n, n]
        for _ in range(n_iter):
            M = M / M.sum(dim=-1, keepdim=True)   # row normalize T_c
            M = M / M.sum(dim=-2, keepdim=True)   # col normalize T_r
        return M

    def get_mappings(self, x: torch.Tensor):
        """
        x: [B, L, n, C]
        """

        B, L, n, C = x.shape
        nC = self.n * C

        # flatten: [B, L, n, C] → [B, L, nC]
        x_vec = x.reshape(B, L, nC)

        # RMSNorm on last dim (nC)
        x_vec_norm = self.norm(x_vec)  # [B, L, nC]

        # 통합 projection: [B, L, nC] @ [nC, n²+2n] → [B, L, n²+2n]
        proj = x_vec_norm @ self.phi.weight.T  # [B, L, n²+2n]

        # split: pre[n], post[n], res[n²]
        proj_pre  = proj[..., :n]           # [B, L, n]
        proj_post = proj[..., n:2*n]        # [B, L, n]
        proj_res  = proj[..., 2*n:]         # [B, L, n²]

        b_pre  = self.b[:n]                 # [n]
        b_post = self.b[n:2*n]             # [n]
        b_res  = self.b[2*n:]              # [n²]

        # H̃ = α * proj + b
        H_tilde_pre  = self.alpha_pre  * proj_pre  + b_pre   # [B, L, n]
        H_tilde_post = self.alpha_post * proj_post + b_post  # [B, L, n]
        H_tilde_res  = self.alpha_res  * proj_res  + b_res   # [B, L, n²]

        # 논문 eq(8): manifold projection
        H_pre  = torch.sigmoid(H_tilde_pre)           # [B, L, n]
        H_post = 2.0 * torch.sigmoid(H_tilde_post)    # [B, L, n]
        H_res  = self.sinkhorn_knopp(
            H_tilde_res.view(B * L, n, n), self.sinkhorn_iter
        ).view(B, L, n, n)                            # [B, L, n, n]

        return H_pre, H_post, H_res

    def forward(self, x: torch.Tensor, sublayer_fn, need_contract:bool=True) -> torch.Tensor:
        """
        논문 eq(3): x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)
        x: [B, L, n, C]
        """
        H_pre, H_post, H_res = self.get_mappings(x)

        if need_contract :
            # H_pre @ x_l: [B,L,n] weighted sum → [B,L,C]
            h_in = (H_pre.unsqueeze(-1) * x).sum(dim=2)           # [B, L, C]
            # sublayer
            h_out = sublayer_fn(h_in)                # [B, L, C]            
        else :
            h_in = (H_pre.unsqueeze(-1) * x)
            # sublayer
            h_out = sublayer_fn(h_in).sum(dim=2)                # [B, L, C]

        # H_res @ x_l: stream mixing
        x_res = torch.einsum('blij,bljc->blic', H_res, x)     # [B, L, n, C]

        # H_post^T @ h_out: expand back
        x_new = x_res + H_post.unsqueeze(-1) * h_out.unsqueeze(2)  # [B, L, n, C]

        return x_new

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            Fast_GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class MSABLock(nn.Module) :
    def __init__(self, emb_dim, n_heads,attn_dropout, ffn_mul, ffn_dropout, use_mhc,hc_mult=4,mask=None):
        super().__init__()
        self.mask = mask
        self.use_mhc = use_mhc

        self.MSA = nn.MultiheadAttention(embed_dim=emb_dim,
                                         num_heads=n_heads,
                                         dropout=attn_dropout,
                                         batch_first=True)
        self.FFN = FeedForwardBlock(emb_size=emb_dim,
                                    expansion=ffn_mul,
                                    drop_p=ffn_dropout)
        
        if use_mhc :
            self.mhc_attn = mHyperConnection(emb_dim, hc_mult, sinkhorn_iter=5)
            self.mhc_ffn = mHyperConnection(emb_dim, hc_mult, sinkhorn_iter=5)
        else :
            self.ln_1 = nn.LayerNorm(normalized_shape=emb_dim)
            self.ln_2 = nn.LayerNorm(normalized_shape=emb_dim)
    def forward(self, x):
        """
        x: [B, L, n, C]  ← mHC stream format
        """
        if self.use_mhc :
            def attn_fn(h):
                # h: [B, L, C]
                out, _ = self.MSA(h, h, h,
                                attn_mask=self.mask,
                                need_weights=False)
                return out

            out = self.mhc_attn(x, attn_fn)          # [B, L, n, C]
            out = self.mhc_ffn(x, self.FFN)          # [B, L, n, C]
        else :
            ln_x = self.ln_1(x)
            attn_out, _ = self.MSA(ln_x, ln_x, ln_x,
                    attn_mask=self.mask,
                    need_weights=False)
            attn_out = attn_out + x
            ffn_out = self.FFN(self.ln_2(attn_out))
            out = ffn_out + attn_out
        return out
    
class MOEBlock(nn.Module):
    def __init__(self, emb_dim, n_heads, attn_dropout, ffn_mul, ffn_dropout,
                 n_experts, k, c, use_mhc, mask=None,hc_mult=4, sinkhorn_iter=20):
        super().__init__()
        self.mask = mask
        self.use_mhc = use_mhc

        self.ln_1 = nn.LayerNorm(normalized_shape=emb_dim)
        self.ln_2 = nn.LayerNorm(normalized_shape=emb_dim)

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

        if use_mhc :
            self.hc_attn = mHyperConnection(emb_dim, hc_mult, sinkhorn_iter)
            self.hc_moe  = mHyperConnection(emb_dim, hc_mult, sinkhorn_iter)

    def forward(self, x: torch.Tensor):
        """x: [B, L, n, C]"""

        if self.use_mhc :
            def attn_fn(h):
                out, _ = self.MSA(h, h, h,
                                    attn_mask=self.mask,
                                    need_weights=False)
                return out

            out = self.hc_attn(x, attn_fn)    # [B, L, n, C]

            # aux_loss를 closure로 캡처
            aux_loss_container = []
            def moe_fn(h):
                out, aux = self.moe(h)
                aux_loss_container.append(aux)
                return out

            out = self.hc_moe(x, moe_fn)      # [B, L, n, C]
            return x, aux_loss_container[0]
        else :
            attn_out, _ = self.MSA(x, x, x,
                    attn_mask=self.mask,
                    need_weights=False)
            attn_out = attn_out + x
            moe_out, aux = self.moe(attn_out)
            out = moe_out + attn_out
            return out, aux

class MSA_Encoder(nn.Module) :
    def __init__(self, emb_dim, n_heads, attn_dropout, ffn_mul, ffn_dropout, depth, hc_mult,use_mhc=False,mask=None, engram_cfg:engram_config=None):
        super().__init__()
        self.engram_cfg = engram_cfg
        self.use_mhc = use_mhc

        self.n_streams = hc_mult

        self.MSA_layers = nn.ModuleList([MSABLock(emb_dim=emb_dim,
                                    n_heads=n_heads,
                                    attn_dropout=attn_dropout,
                                    ffn_mul=ffn_mul,
                                    ffn_dropout=ffn_dropout,
                                    mask=mask,
                                    use_mhc=use_mhc,
                                    hc_mult=hc_mult) for _ in range(depth)])
        if engram_config.engram_layer_n :
            if use_mhc :
                self.engram_layer = nn.ModuleList([EngramModule(engram_cfg, hc_mult) for _ in engram_config.engram_layer_n])
                self.engram_mhc = nn.ModuleList([mHyperConnection(emb_dim, hc_mult, sinkhorn_iter=20) for _ in engram_config.engram_layer_n])
            else :
                self.engram_layer = nn.ModuleList([EngramModule(engram_cfg, 1) for _ in engram_config.engram_layer_n])

    def forward(self, x, engram_embedding_table=None, engram_token_id=None) :
        out = x
        if len(x.shape) == 3 and self.use_mhc:
            out.unsqueeze_(2)
            out = out.expand(-1, -1, self.n_streams, -1)

        for idx, layer in enumerate(self.MSA_layers) :
            if idx + 1 in self.engram_cfg.engram_layer_n :
                layer_idx = self.engram_cfg.engram_layer_n.index(idx+1)
                if self.use_mhc :
                    def engram_fn(h) :
                        out = self.engram_layer[layer_idx](h, engram_token_id, engram_embedding_table[layer_idx])
                        return out
                    
                    out = self.engram_mhc[layer_idx](out, engram_fn, need_contract=False)
                    out = layer(out)
                else :
                    out = out + self.engram_layer[layer_idx](x, engram_token_id, engram_embedding_table[layer_idx])
            else :
                out = out + layer(out)
        return out

    
class MOE_Encoder(nn.Module) :
    def __init__(self,emb_dim, n_heads, attn_dropout, ffn_mul, ffn_dropout, c, k, n_experts,depth, every_2, hc_mult,use_mhc=False,mask=None, engram_cfg:engram_config=None):
        super().__init__()
        self.engram_cfg = engram_cfg
        self.n_streams = hc_mult
        self.use_mhc = use_mhc

        if every_2 :
            self.MOE_layer = nn.ModuleList([layer for _ in range(depth//2) for layer in (
                            MSABLock(emb_dim=emb_dim,
                                        n_heads=n_heads,
                                        attn_dropout=attn_dropout,
                                        ffn_mul=ffn_mul,
                                        hc_mult=hc_mult,
                                        ffn_dropout=ffn_dropout,
                                        use_mhc=use_mhc,
                                        mask=mask),

                            MOEBlock(emb_dim=emb_dim,
                                        n_heads=n_heads,
                                        attn_dropout=attn_dropout,
                                        ffn_mul=ffn_mul,
                                        ffn_dropout=ffn_dropout,
                                        c=c,
                                        k=k,
                                        n_experts=n_experts,
                                        hc_mult=hc_mult,
                                        use_mhc=use_mhc,
                                        mask=mask))])
        else :
            self.MOE_layer = nn.ModuleList([MSABLock(emb_dim=emb_dim,
                                    n_heads=n_heads,
                                    attn_dropout=attn_dropout,
                                    ffn_mul=ffn_mul,
                                    hc_mult=hc_mult,
                                    use_mhc=use_mhc,
                                    ffn_dropout=ffn_dropout, mask=mask) for _ in range(depth-2)])
            for _ in range(2) :
                self.MOE_layer.append(MOEBlock(emb_dim=emb_dim,
                                        n_heads=n_heads,
                                        attn_dropout=attn_dropout,
                                        ffn_mul=ffn_mul,
                                        ffn_dropout=ffn_dropout,
                                        c=c,
                                        k=k,
                                        n_experts=n_experts,
                                        hc_mult=hc_mult,
                                        use_mhc=use_mhc,
                                        mask=mask))
        if engram_config.engram_layer_n :
            if use_mhc :
                self.engram_layer = nn.ModuleList([EngramModule(engram_cfg, n_streams=hc_mult) for _ in engram_config.engram_layer_n])
                self.engram_mhc = nn.ModuleList([mHyperConnection(emb_dim, hc_mult, sinkhorn_iter=20)])
            else :
                self.engram_layer = nn.ModuleList([EngramModule(engram_cfg, n_streams=1) for _ in engram_config.engram_layer_n])

    def forward(self, x, engram_embedding_table=None, engram_token_id=None) :

        out = x
        aux_loss = 0
        if len(x.shape) == 3 and self.use_mhc:
            out.unsqueeze_(2)
            out = out.expand(-1, -1, self.n_streams, -1)

        for idx, layer in enumerate(self.MOE_layer) :
            if idx + 1 in self.engram_cfg.engram_layer_n :
                layer_idx = self.engram_cfg.engram_layer_n.index(idx+1)
                if self.use_mhc :
                    def engram_fn(h, _idx=layer_idx):
                        return self.engram_layer[_idx](h, engram_token_id, engram_embedding_table[_idx])

                    out = self.engram_mhc[layer_idx](out, engram_fn, need_contract=False)
                else :
                    out = self.engram_layer[layer_idx](out, engram_token_id, engram_embedding_table[layer_idx])

            if layer._get_name() != "MOEBlock" :
                out = layer(out)
            else :
                out, loss = layer(out)
                aux_loss += loss
        
        return out, aux_loss

if __name__ == "__main__" :
    test_model = MOE_Encoder(512, 4, 0.1, 1, 0.1, 1, 1, 16, 4, False, hc_mult=4,use_mhc=True, engram_cfg=engram_config).to(device)
    test_model_2 = MSA_Encoder(512, 4, 0.1, 1, 0.1, 16,use_mhc=True,hc_mult=4, engram_cfg=engram_config).to(device)

    embedding_list = nn.ModuleList([nn.Embedding(sum([engram_config.engram_vocab_size] * 2) * 2, engram_config.engram_embd_d, device=device),
                                    nn.Embedding(sum([engram_config.engram_vocab_size] * 2) * 2, engram_config.engram_embd_d, device=device)])
    
    input_data_l = [torch.randn((10, 32, 512), device=device), embedding_list, torch.randint(0, 10, (10, 32), device=device)]

    print(test_model(torch.randn((10, 32, 512), device=device), embedding_list, torch.randint(0, 10, (10, 32), device=device))[0].shape)
    print(test_model_2(torch.randn((10, 32, 512), device=device), embedding_list, torch.randint(0, 10, (10, 32), device=device)).shape)
