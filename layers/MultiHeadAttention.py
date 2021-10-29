import random

import torch
from torch import nn
from einops import rearrange
import copy
from layers import Head


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads=8, in_dim=256, head_dim=64, out_dim=256):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.projection = nn.Linear(num_heads * head_dim, out_dim)
        self.heads = nn.ModuleList([
            Head(head_dim, head_dim) for i in range(num_heads)
        ])
        self.multi_head_count = 8

    def forward(self, x, require_attention=False):
        head_output = []
        attn_output = []
        x = rearrange(x, 'b n (h d) -> b h n d', h=self.multi_head_count)
        for (i, head) in enumerate(self.heads):
            if require_attention:
                per_head,attn = head(x[:, i],require_attention=True)
                attn_output.append(attn)
            else:
                per_head = head(x[:, i])
            head_output.append(per_head)
        out = torch.stack(head_output, dim=1)
        if require_attention:
            attn = torch.stack(attn_output,dim=1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        if require_attention:
            print(attn.shape)
            return out,attn
        return out

    def increaseHead(self):
        self.num_heads += 1

        # Copy old projection weights to the new projection
        # old_projection = self.projection
        # self.projection = nn.Linear(self.num_heads * self.head_dim, self.out_dim)
        # self.projection.weight.data[:, :(self.num_heads - 1) * self.head_dim] = old_projection.weight.data
        # self.projection.bias.data = old_projection.bias.data
        new_head = Head(self.head_dim, self.head_dim)
        new_head.apply(MultiHeadAttention.init_weight)
        # Freeze the old heads
        # for head in self.heads:
        #     head.requires_grad_(False)
        self.heads.append(new_head)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
