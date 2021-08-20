from torch import nn

from layers import TransformerEncoderBlock


class TransformerEncoder(nn.Module):
    def __init__(self, depth=12):
        super(TransformerEncoder, self).__init__()
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock() for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
