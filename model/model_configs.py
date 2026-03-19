from Image_encoder import ViTConfig
from dataclasses import dataclass

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
            use_moe=True,
            every_2=False
    )
    VIT_B_32 = ViTConfig(
            emb_dim=768,
            n_heads=12,
            ffn_mul=4,
            n_experts=32,
            depth=12,
            patch_size=32,
            use_moe=False,
    )
    VMOE_B_32 = ViTConfig(
            emb_dim=768,
            n_heads=12,
            ffn_mul=4,
            n_experts=32,
            depth=12,
            patch_size=32,
            c=1.05,
            k=5,
            use_moe=True,
            every_2=False
    )
