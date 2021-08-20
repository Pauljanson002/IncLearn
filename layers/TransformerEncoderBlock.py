from torch import nn
from layers import MultiHeadAttention, FeedForwardBlock


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_dim=256, num_heads=1, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.pre_norm = nn.LayerNorm(emb_dim)
        self.mha = MultiHeadAttention(num_heads=num_heads,in_dim=256,head_dim=32,out_dim=256)
        self.dropout1 = nn.Dropout(dropout)
        self.ff = FeedForwardBlock(emb_size=emb_dim)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.mha(self.pre_norm(x)))
        x = x + self.dropout2(self.ff(self.norm1(x)))
        return x
