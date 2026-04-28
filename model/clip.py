import torch
from torch import Tensor, nn
import numpy as np

from model_configs import ViTConfig, TETConfig, engram_config_set

from Image_encoder import VIT
from text_encoder import TET
from engram import engram_config

device = "cuda" if torch.cuda.is_available() else "cpu"

class CLIP(nn.Module) :
    def __init__(self, image_encoder_cfg:ViTConfig, text_encoder_cfg:TETConfig):
        super().__init__()
        self.i_cfg = image_encoder_cfg
        self.t_cfg = text_encoder_cfg

        self.n_ctx = TETConfig.max_ctx_len

        self.image_encoder = VIT(self.i_cfg)
        self.text_encoder = TET(self.t_cfg)

        self.ln_t = nn.LayerNorm(self.t_cfg.emb_dim)
        self.ln_i = nn.LayerNorm(self.i_cfg.emb_dim)
    
        self.text_proj = nn.Parameter(torch.empty(self.t_cfg.emb_dim, self.t_cfg.emb_dim))
        self.img_proj = nn.Parameter(torch.empty(self.i_cfg.emb_dim, self.t_cfg.emb_dim))

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if text_encoder_cfg.engram_cfg :
            self.engram_embedding = nn.ModuleList([
                nn.Embedding((text_encoder_cfg.engram_cfg.engram_vocab_size * len(range(text_encoder_cfg.engram_cfg.max_ngram - 1))) * 2, text_encoder_cfg.engram_cfg.engram_embd_d) for _ in text_encoder_cfg.engram_cfg.engram_layer_n
            ])

    @property
    def initialize_parameters(self) :
        nn.init.normal_(self.text_encoder.token_emb.embedding.weight, std=0.02)
        nn.init.normal_(self.text_encoder.token_emb.pos_emb, std=0.01)

        i_std = (self.i_cfg.emb_dim ** -0.5) * ((2 * self.i_cfg.emb_dim) ** -0.5)
        for m in self.image_encoder.parameters() :
            nn.init.normal_(m, std=i_std)

        t_std = (self.t_cfg.emb_dim ** -0.5) * ((2 * self.t_cfg.emb_dim) ** -0.5)
        for m in self.text_encoder.parameters() :
            nn.init.normal_(m, std=t_std)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.t_cfg.emb_dim, self.t_cfg.emb_dim)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def encode_text(self, text) :
        feat = self.text_encoder(text, self.engram_embedding).sum(dim=2)
        feat = self.ln_t(feat)
        if self.image_encoder
        output = feat[torch.arange(feat.shape[0]), text.argmax(dim=-1)] @ self.text_proj
        return output
        
    def encode_image(self, image) :
        feat = self.image_encoder(image, self.engram_embedding).sum(dim=2)
        feat = self.ln_i(feat)
        output = feat[:, 0, :] @ self.img_proj
        return output

    def forward(self, data) :
        if self.i_cfg.use_moe :
            image_feature, i_aux_loss = self.encode_image(data[0])
        else :
            image_feature = self.encode_image(data[0])

        if self.t_cfg.use_moe :
            text_feature, t_aux_loss = self.encode_text(data[1])
        else :
            text_feature = self.encode_text(data[1])

        #norm
        image_feature = image_feature / image_feature.norm(dim=1, keepdim=True)
        text_feature = text_feature / text_feature.norm(dim=1, keepdim=True)

        #cos sim
        logit_scale = self.logit_scale.exp()
        logit_per_image = logit_scale * image_feature @ text_feature.t()
        logit_per_text = logit_per_image.t()

        return logit_per_image, logit_per_text, (t_aux_loss if self.t_cfg.use_moe else None), (i_aux_loss if self.i_cfg.use_moe else None)

if __name__ == "__main__" :
    import model_configs
    import torchinfo

    temp_model = CLIP(image_encoder_cfg = model_configs.vit_model.VIT_S_32,
                      text_encoder_cfg=model_configs.tet_model.TET_S_32).to(device)
    test_img = torch.randn((100, 3,224, 224), device=device)
    test_text = torch.randint(0, 100, (100, 77), device=device)

    l_i, l_t, _, _ = temp_model([test_img, test_text])
    print(l_i.shape)
    print(l_t.shape)
