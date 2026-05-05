from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn

class EfficientNetEncoder(nn.Module):
    def __init__(self,
                 in_channels: int = 2
                 ) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.model = EfficientNet.from_name("efficientnet-b0", include_top=False, in_channels=self.in_channels)

        self.model._avg_pooling = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)