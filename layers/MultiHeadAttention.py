import torch
from torch import nn
from einops import rearrange
import copy
from layers import Head


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads=1, in_dim=256, head_dim=256, out_dim=256):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.projection = nn.Linear(head_dim, out_dim)
        self.heads = nn.ModuleList([
            Head(in_dim, head_dim) for i in range(num_heads)
        ])

    def forward(self, x):
        head_output = 0
        for head in self.heads:
            head_output += head(x)
        out = head_output
        # out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        return out

    def increaseHead(self):
        self.num_heads += 1

        # Copy old projection weights to the new projection
        # old_projection = self.projection
        # self.projection = nn.Linear(self.num_heads * self.head_dim, self.out_dim)
        # self.projection.weight.data[:, :(self.num_heads - 1) * self.head_dim] = old_projection.weight.data
        # self.projection.bias.data = old_projection.bias.data
        new_head = copy.deepcopy(self.heads[-1])
        # Freeze the old heads
        for head in self.heads:
            head.requires_grad_(False)
        self.heads.append(new_head)
