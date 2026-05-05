import torch
import torch.nn as nn

class LearnedFeaturesSpectralPooling(nn.Module):
    def __init__(self, proj, out_channels=80, output_components=8, complex_output=False, norm=None, center_padding=True):
        super().__init__()
        self.project_in = proj
        self.out_channels = out_channels
        self.output_components = output_components
        self.complex_output = complex_output
        self.norm = norm
        self.center_padding = center_padding
    
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(self, x):
        # input will be of shape B, C, F, T
        x = x.transpose(1, 3) # become B, T, F, C
        x = self.project_in(x).transpose(-1, -2) # become B, d, T
        
        # Learnable temporal compression
        offset = max(0, (self.output_components+1) * 2 + 1 - x.shape[-1])
        if self.center_padding:
            l_pad = offset // 2
            r_pad = offset - l_pad
        else:
            l_pad, r_pad = 0, offset

        r = nn.functional.pad(x, (l_pad, r_pad))

        # FFT over the temporal axis
        r = torch.fft.rfftn(r, dim=(-1,), norm='ortho') # (B, N, T//2 + 1), 'ortho' norm gives invariance to input length

        assert r.shape[-1] >= (self.output_components + 1), f"Expected FFT components ({self.output_components} + 1) exceed available frequency bins ({r.shape[-1]}). Reduce output_components."
        
        r = r[..., 1:self.output_components+1]

        mag = r.abs()#**2
        match self.norm:
            case 'component_wise':
                scale = mag.mean(dim=-2, keepdim=True)
            case 'channel_wise':
                scale = mag.mean(dim=-1, keepdim=True)
            case 'global':
                scale = mag.mean(dim=(-2, -1), keepdim=True)
            case _:
                scale = 1
        scale = (scale + 1e-6) #** .5
        r = r / scale

        if self.complex_output:
            return torch.stack([r.real, r.imag], dim=1)
        else:
            return r.abs()