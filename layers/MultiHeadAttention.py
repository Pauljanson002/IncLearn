import random

import torch
from torch import nn
from einops import rearrange
import copy
from layers import Head


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads=4, in_dim=256, head_dim=64, out_dim=256):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.projection = nn.Linear(num_heads*head_dim, out_dim)
        self.heads = nn.ModuleList([
            Head(head_dim, head_dim) for i in range(num_heads)
        ])
        self.multi_head_count = 4

    def forward(self, x):
        head_output = torch.tensor([],device='cuda')
        x = rearrange(x,'b n (h d) -> b h n d',h=self.multi_head_count)
        random_head_list = sorted(random.sample(range(0,self.num_heads),self.multi_head_count))
        # print("Heads selected")
        # print(random_head_list)
        for (i,head_loc) in enumerate(random_head_list):
            head = self.heads[head_loc]
            per_head = head(x[:,i])
            head_output = torch.cat([head_output,per_head],dim=-1)
        out = head_output
        #out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
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
