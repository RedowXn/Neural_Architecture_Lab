# src/model.py
import torch
import torch.nn as nn

class TinyTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
    def forward(self, x):
        # x shape: (seq, batch, features)
        return self.encoder(x)
