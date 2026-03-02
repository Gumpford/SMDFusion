import torch
import torch.nn as nn
from .classifier import FrozenClassifier
from .dino_encoder import DinoEncoder
from .prompt_pool import PromptPool
from .gated_adapter import GatedAdapter
from .recon_decoder import ReconDecoder


class ReconModel(nn.Module):
    def __init__(self, classifier_ckpt: str, dino_model_name="dinov2_vitb14", hidden_dim=256):
        super().__init__()
        self.classifier = FrozenClassifier(classifier_ckpt)
        self.encoder = DinoEncoder(dino_model_name, hidden_dim)
        self.prompt_pool = PromptPool(hidden_dim)
        self.adapters = nn.ModuleList([GatedAdapter(hidden_dim) for _ in range(4)])
        self.decoder = ReconDecoder(hidden_dim, out_ch=3)

    def freeze_backbones(self):
        for m in [self.classifier, self.encoder.model]:
            for p in m.parameters():
                p.requires_grad = False

    def set_prompt_trainable(self, flag: bool):
        for p in self.prompt_pool.parameters():
            p.requires_grad = flag
        for a in self.adapters.parameters():
            a.requires_grad = flag

    def forward(self, x_deg: torch.Tensor, enable_prompt: bool = True, temperature: float = 1.0):
        cls = self.classifier(x_deg)
        z = self.prompt_pool.compose(cls["p_mod"].detach(), cls["p_deg_ir"].detach(), cls["p_deg_vi"].detach(), temperature)
        feats = self.encoder.extract_multi_layer_feats(x_deg)
        out_feats, gates = [], []
        for f, ad in zip(feats, self.adapters):
            fo, g = ad(f, z, enable_prompt=enable_prompt)
            out_feats.append(fo)
            gates.append(g)
        x_hat = self.decoder(out_feats)
        return x_hat, z, gates, cls
