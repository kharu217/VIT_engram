from Image_encoder import ViTConfig
from text_encoder import TETConfig
from engram import engram_config
from dataclasses import dataclass
import torch.optim as optim


#small model need to use 362 output_dim
@dataclass
class vit_model :
    VIT_S_32 = ViTConfig(
            emb_dim=512,
            n_heads=8,
            ffn_mul=4,
            depth=8,
            patch_size=32,
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


class tet_model :
    TET_S_32 = TETConfig(
        depth=8,
        emb_dim=256,
        n_heads=8
    )

    TMOE_S_32 = TETConfig(
        depth=8,
        emb_dim=256,
        n_heads=8,
        n_experts=32,
        use_moe=True,
        every_2=False,
        c=1.05,
        k=1,
    )
    TET_B_32 = TETConfig(
        emb_dim=512,
        n_heads=8,
        ffn_mul=4,
        n_experts=32,
        depth=12,
        use_moe=False,
    )

    TMOE_B_32 = TETConfig(
        emb_dim=512,
        n_heads=8,
        ffn_mul=4,
        n_experts=32,
        depth=12,
        c=1.05,
        k=5,
        use_moe=True,
        every_2=False
    )

class engram_config_set :
    engram_config_text_S = engram_config(
        embd_d=256,
        engram_vocab_size=20000,
        max_ngram=3,
        engram_embd_d=256,
        vocab_size=49408
    )

    engram_config_image_S = engram_config(
        embd_d=512,
        engram_vocab_size=20000,
        max_ngram=3,
        engram_embd_d=256,
        vocab_size=49408
    )

    engram_config_image_B = engram_config(
        embd_d=512,
        engram_vocab_size=20000,
        max_ngram=3,
        engram_embd_d=256,
        vocab_size=49408
    )

    engram_config_image_B = engram_config(
        embd_d=512,
        engram_vocab_size=20000,
        max_ngram=3,
        engram_embd_d=256,
        vocab_size=49408
    )

if __name__ == "__main__" :
    from text_encoder import TET
    from Image_encoder import VIT
    import torchinfo
    import torch

    test_image = VIT(cfg=vit_model.VIT_B_32)
    vit_sum = torchinfo.summary(test_image, (10, 3, 224, 224), verbose=0, mode="train")
    estimated_total = (
        vit_sum.total_input +
        vit_sum.total_output_bytes+
        vit_sum.total_param_bytes
    )
    print(f"{round(estimated_total/1e6, 2) * 2}MB, VIT total vram usage(when using adam)")
    print(f"total {round(vit_sum.total_params/1e6, 1)}M")

    print("----------------------------------------")

    test_text = TET(cfg=tet_model.TET_B_32)
    test_data = torch.randint(0, 49407, (10, 77))
    tet_sum = torchinfo.summary(test_text, input_data=test_data, verbose=0, mode="train")
    estimated_total = (
        tet_sum.total_input +
        tet_sum.total_output_bytes+
        tet_sum.total_param_bytes
    )
    print(f"{round(estimated_total/1e6, 2) * 2}MB, TET total vram usage(when using adam)")
    print(f"total {round(tet_sum.total_params/1e6, 1)}M")
