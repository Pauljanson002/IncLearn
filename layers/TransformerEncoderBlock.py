from torch import nn
from layers import MultiHeadAttention, FeedForwardBlock
from layers.DropPath import DropPath


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_dim=128, num_heads=8, dropout=0.1, drop_path_rate=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.pre_norm = nn.LayerNorm(emb_dim)
        self.mha = MultiHeadAttention(num_heads=num_heads, in_dim=128, head_dim=16, out_dim=128)
        self.dropout1 = nn.Dropout(dropout)
        self.ff = FeedForwardBlock(emb_size=emb_dim)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x,require_attention=False):
        if require_attention:
            out, attn = self.mha(self.pre_norm(x),require_attention=True)
        else:
            out = self.mha(self.pre_norm(x))
        x = x + self.drop_path(out)
        x = self.norm1(x)
        x_2 = self.ff(x)
        x = x + self.drop_path(self.dropout2(x_2))
        if require_attention:
            return  x, attn
        return x
