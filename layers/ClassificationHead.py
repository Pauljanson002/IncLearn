import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


class ClassificationHead(nn.Module):
    def __init__(self, emb_size=256, n_classes=100, convolution=False):
        super(ClassificationHead, self).__init__()
        self.layer_norm = nn.LayerNorm(emb_size)
        self.linear = nn.Linear(emb_size, n_classes)
        self.convolution = convolution
        if convolution:
            self.attention_pool = nn.Linear(emb_size, 1)

    def forward(self, x):
        if not self.convolution:
            x = x[:, 0]
        else:
            out = F.softmax(self.attention_pool(x), dim=1)
            x = torch.einsum('bnt,bne -> bte', out, x)
            x = rearrange(x, 'b t e -> b (t e)')
        x = self.linear(self.layer_norm(x))
        return x
