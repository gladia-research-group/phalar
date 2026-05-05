import torch
import torch.nn as nn

class Scale(nn.Module):
    def __init__(self, dim, pos=1):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.pos = pos

    def forward(self, x):
        # x of shape B, dim, *
        view_shape = [1] * x.ndim
        view_shape[self.pos] = -1
        return x * self.weight.view(*view_shape)
