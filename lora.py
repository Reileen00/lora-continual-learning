import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(base.in_features, r)*0.01)
        self.B = nn.Parameter(torch.zeros(r, base.out_features))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A @ self.B) * self.scale
