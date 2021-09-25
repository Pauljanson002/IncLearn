from torch import nn
import torch
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, emb_dim=256,head_dim= 256, dropout=0.1):
        super(Head, self).__init__()
        self.head_dim = head_dim
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.att_drop = nn.Dropout(dropout)
        self.scaling = self.head_dim ** -0.5

    def forward(self, x):
        queries = self.queries(x)
        keys = self.keys(x)
        values = self.values(x)
        energy = torch.einsum('bqd,bkd -> bqk', queries, keys)
        att = F.softmax(energy, dim=-1) * self.scaling
        att = self.att_drop(att)
        out = torch.einsum('bal,blv -> bav', att, values)
        return out

