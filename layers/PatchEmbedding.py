import torch

from torch import nn
from einops import repeat
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=3, emb_size=256, img_size=32, convolution=False):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.sequence_length = (img_size // 2) ** 2
        self.convolution = False
        if convolution:
            self.convolution = True
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, emb_size, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                # nn.Conv2d(emb_size, emb_size, kernel_size=2, stride=1, padding=1, bias=False),
                # nn.ReLU(),
                # nn.MaxPool2d(kernel_size=2)
            )
        else:
            self.sequence_length += 1
            self.projection = nn.Conv2d(
                in_channels=in_channels,
                out_channels=emb_size,
                kernel_size=patch_size,
                stride=patch_size
            )
        self.rearrange = Rearrange('b e (h) (w) -> b (h w) e')
        if not convolution:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size), requires_grad=True)
        self.positional_emb = nn.Parameter(PatchEmbedding.sinusoidal_embedding(self.sequence_length, emb_size),
                                           requires_grad=False)

        self.apply(PatchEmbedding.init_weight)

    def forward(self, x):
        x = self.projection(x)
        x = self.rearrange(x)
        batch_size = x.shape[0]
        if not self.convolution:
            cls_token_repeated = repeat(self.cls_token, '() n e -> b n e', b=batch_size)
            x = torch.cat([cls_token_repeated, x], dim=1)
        x = x + self.positional_emb
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)
