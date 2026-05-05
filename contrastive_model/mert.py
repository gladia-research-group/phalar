import torch
import torch.nn as nn
import torchaudio.transforms as T
from transformers import AutoModel, Wav2Vec2FeatureExtractor

class MERTEncoder(nn.Module):
    def __init__(self, 
                 input_sr: int = 16000, 
                 model_id: str = "m-a-p/MERT-v1-95M", 
                 layer_idx: int = -1, # -1 means the last layer
                 ):
        super().__init__()

        # raise NotImplementedError("MERTEncoder has been disabled to avoid transformers dependency.")
        
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id,trust_remote_code=True)
        self.output_dim = self.model.config.hidden_size
        
        total_layers = self.model.config.num_hidden_layers
        if layer_idx < 0:
            self.layer_idx = total_layers + 1 + layer_idx
        else:
            self.layer_idx = layer_idx

        # Prune the encoder layers to stop inference early
        # If we need layer 'n', we only need n transformer blocks
        if self.layer_idx > 0 and self.layer_idx <= total_layers:
            self.model.encoder.layers = self.model.encoder.layers[:self.layer_idx]
        
        self.model.eval()
        self.model.requires_grad_(False)

        self.target_sr = self.processor.sampling_rate
        self.input_sr = input_sr
        if input_sr != self.target_sr:
            self.resampler = T.Resample(input_sr, self.target_sr)
        else:
            self.resampler = nn.Identity

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resampler(x)
        og_dtype = x.dtype
        x = x.float()
        
        # MERT expects (Batch, Time)
        if x.shape[1] != 1:
            x = x.mean(1, keepdim=True)

        x = x[:, 0]

        outputs = self.model(input_values=x, output_hidden_states=True)
        
        # Select requested layer
        h = outputs.hidden_states[-1]  # B, T, d
        
        return h.permute(0, 2, 1).unsqueeze(2)