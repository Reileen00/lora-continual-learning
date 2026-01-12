import torch.nn as nn
from lora import LoRALinear

class LoRATransformer(nn.Module):
    def __init__(self, vocab=1000, dim=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        layer = nn.TransformerEncoderLayer(dim,4)
        self.tr = nn.TransformerEncoder(layer,4)
        self.fc = LoRALinear(nn.Linear(dim, vocab))

    def forward(self,x):
        x = self.emb(x)
        x = self.tr(x)
        return self.fc(x)
