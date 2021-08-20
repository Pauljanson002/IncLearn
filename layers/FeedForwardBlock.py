from torch import nn


class FeedForwardBlock(nn.Module):
    def __init__(self, emb_size=256, expansion=2, drop_p=0.1):
        super(FeedForwardBlock, self).__init__()
        self.expander = nn.Linear(emb_size, expansion * emb_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(drop_p)
        self.reducer = nn.Linear(expansion * emb_size, emb_size)

    def forward(self, x):
        out = self.expander(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.reducer(out)
        return out
