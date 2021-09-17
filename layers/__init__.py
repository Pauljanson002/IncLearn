from .Head import Head
from .MultiHeadAttention import MultiHeadAttention
from .FeedForwardBlock import FeedForwardBlock
from .TransformerEncoderBlock import TransformerEncoderBlock
from .TransformerEncoder import TransformerEncoder
from .PatchEmbedding import PatchEmbedding
from .ClassificationHead import ClassificationHead
from .DropPath import DropPath
import torch
from torch import nn


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x):
        return self.alpha * x + self.beta
