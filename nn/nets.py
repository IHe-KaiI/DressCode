import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

SOS = 2001
EOS = 2002
PAD = 2003

# ------ Basic Interface --------
class BaseModule(nn.Module):
    """Base interface for my neural nets"""
    def __init__(self):
        super().__init__()
        self.config = {
            'model': self.__class__.__name__
        }
    
    def train(self, mode=True):
        super().train(mode)
    
    def eval(self):
        super().eval()

class SewingGPT(BaseModule):
    def __init__(self):
        super().__init__()

        n_grid = SOS
        dim = emb_dim = 512
        clip_feature_txt = 1024
        clip_feature_img = clip_feature_pc = 768
        heads = 8
        depth = 24
        vocab_size = 3 + n_grid

        self.value_emb = nn.Embedding(vocab_size, emb_dim) 
        self.axis_emb = nn.Embedding(120, emb_dim) 
        self.pos_emb = nn.Embedding(30, emb_dim)

        self.proj_in = nn.Linear(emb_dim, dim, bias=False)

        self.proj_feature_txt = nn.Linear(clip_feature_txt, dim, bias=False)

        self.layers = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, heads, dim_feedforward=dim * 4, dropout=0, activation=F.gelu, batch_first=True, norm_first=True),
            depth
        )

        self.proj_out = nn.Linear(dim, vocab_size, bias=False)

        self.dim = dim
        self.vocab_size = vocab_size

        print(f"numel = {sum([i.numel() for i in self.parameters()])/1e6:.1f} M parameters")

    def forward(self, indices_value, indices_axis, indices_pos, feature):
        
        x = self.value_emb(indices_value) + self.axis_emb(indices_axis) + self.pos_emb(indices_pos)
        x = x * np.sqrt(self.dim)
        
        x = self.proj_in(x)
        x = self.layers(x, feature, tgt_mask=nn.Transformer.generate_square_subsequent_mask(x.shape[1], x.device))
        x = self.proj_out(x)

        return x


