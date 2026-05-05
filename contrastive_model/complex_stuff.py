import torch
import torch.nn as nn

class CplxLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.W = nn.Linear(in_f, out_f*2, bias=False)

        nn.init.kaiming_uniform_(self.W.weight, a=5**0.5)
        with torch.no_grad():
            self.W.weight.mul_(1 / 2**0.5)

        self.bias = bias
        if self.bias:
            self.br = nn.Parameter(torch.zeros(out_f))
            self.bi = nn.Parameter(torch.zeros(out_f))

    def forward(self, x):  # x: complex, real/imag in half
        rp, ip = torch.chunk(self.W(x), 2, dim=-1)

        yr = (rp[:, 0] - ip[:, 1])
        yi = (rp[:, 1] + ip[:, 0])

        if self.bias:
            yr = yr + self.br[None].to(yr.dtype)
            yi = yi + self.bi[None].to(yi.dtype)

        return torch.stack([yr, yi], dim=1)

class CplxRMSNorm(nn.Module):
    def __init__(
        self,
        channels: int
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is B 2 C
        # |x|^2 = Re^2 + Im^2
        mag_sq = (x**2).sum(dim=1)  # B C

        scale = torch.sqrt(mag_sq.mean(dim=-1, keepdim=True) + 1e-6)[:, None]

        # normalization
        x_norm = x / scale

        return x_norm * self.weight.view(1, 1, -1).to(x.dtype)

class modReLU(nn.Module):
    def __init__(self, channels):
        super(modReLU, self).__init__()
        self.b = nn.Parameter(torch.Tensor(channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.b, -0.01)

    def forward(self, z):
        mag = ((z**2).sum(dim=1) + 1e-6) ** .5
        
        bias = self.b.view(1, -1)
        
        scale = torch.relu(mag + bias) / mag
        
        return z * scale[:, None]

class CplxSiLU(nn.Module):
    def __init__(self, channels):
        super(CplxSiLU, self).__init__()
        self.b = nn.Parameter(torch.Tensor(channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.b, -0.01)

    def forward(self, z):
        mag = ((z**2).sum(dim=1) + 1e-6) ** .5
        
        bias = self.b.view(1, -1)
        
        scale = (mag + bias) * torch.sigmoid(mag + bias) / mag
        
        return z * scale[:, None]

class CplxDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        # input is B 2 *

        # total magnitude is decreased to m * (1 - self.p)
        mask = (torch.rand(tuple(v for i, v in enumerate(x.shape) if i!=1), device=x.device) > self.p).to(x.dtype)
        mask = mask / (1 - self.p)

        return x * mask[:, None]