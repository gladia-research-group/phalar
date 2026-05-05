import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from contrastive_model.utils import Scale

class FreqTimeSeparableResBlock(nn.Module):
    def __init__(self, channels, out_channels, dropout=0.1, compress=True):
        super().__init__()

        # freq_conv should act on B*T, C, F
        self.freq_conv = nn.Sequential(
            nn.GroupNorm(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            Scale(channels),
            nn.Dropout2d(p=dropout, inplace=True)
        )

        # init convs
        nn.init.kaiming_normal_(self.freq_conv[1].weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_normal_(self.freq_conv[3].weight)

        nn.init.zeros_(self.freq_conv[1].bias)

        # time_conv should act on B*F, C, T
        self.time_conv = nn.Sequential(
            nn.GroupNorm(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=(1,3), padding=(0,1)),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            Scale(channels),
            nn.Dropout2d(p=dropout, inplace=True)
        )

        # init convs
        nn.init.kaiming_normal_(self.time_conv[1].weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_normal_(self.time_conv[3].weight)

        nn.init.zeros_(self.time_conv[1].bias)

        # pointwise op act on B*T, C, F
        self.pointwise_op = nn.Sequential(
            nn.GroupNorm(channels, channels),
            nn.Conv2d(channels, 2*channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(2*channels, channels, kernel_size=1, bias=False),
            Scale(channels),
            nn.Dropout2d(p=dropout, inplace=True)
        )

        # init convs
        nn.init.kaiming_normal_(self.pointwise_op[1].weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_normal_(self.pointwise_op[3].weight)

        nn.init.zeros_(self.pointwise_op[1].bias)
        
        if compress:
            self.compression = nn.Conv2d(channels, out_channels, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), bias=False)
            nn.init.xavier_normal_(self.compression.weight)
        else:
            self.compression = nn.Identity()

    def forward(self, x):
        x = x + self.freq_conv(x)
        x = x + self.time_conv(x)
        x = x + self.pointwise_op(x)
        return self.compression(x)

class FreqTimeSeparableEncoder(nn.Module):
    def __init__(self, in_channels=1, init_hidden_dim=8, depth=4, dropout_p=0.1, use_checkpointing=True):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        # Initial projection: Produce channels
        self.input_proj = nn.Conv2d(in_channels, init_hidden_dim, kernel_size=1, bias=False)
        nn.init.xavier_normal_(self.input_proj.weight)

        layers = []
        dims = [init_hidden_dim]
        for _ in range(depth):
            dims.append(int(dims[-1]*1.5))
            layers.append(FreqTimeSeparableResBlock(channels=dims[-2], out_channels=dims[-1], dropout=dropout_p, compress=True))
            layers.append(FreqTimeSeparableResBlock(channels=dims[-1], out_channels=dims[-1], dropout=dropout_p, compress=False))
        
        self.layers = nn.ModuleList(layers)
        self.output_dim = dims[-1]
    
    def forward(self, x):
        x = self.input_proj(x)
        
        for layer in self.layers:
            if self.use_checkpointing and self.training: # Do this to save space during training!
                # What this does is NOT storing the inner feature-maps of this call, but only the input,
                #   then, during the backward pass, it recomputes the forward just for that point.
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        return x