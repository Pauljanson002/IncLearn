from torch import nn
import torch
from layers import TransformerEncoderBlock


class TransformerEncoder(nn.Module):
    def __init__(self, depth=7, dropout_rate=0.1, stochastic_depth_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        dpr = [i.item() for i in torch.linspace(0, stochastic_depth_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(drop_path_rate=dpr[_]) for _ in range(depth)
        ])
        self.apply(self.init_weight)

    def forward(self, x):
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
