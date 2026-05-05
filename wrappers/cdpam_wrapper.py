import os
import torch
import torch.nn.functional as F
import lightning as L
import torchaudio
from cdpam.models import FINnet
import numpy as np
from functools import partial

class CDPAMComparisonPLWrapper(L.LightningModule):
    def __init__(
        self,
        modfolder='.hidden/cdpam_scratchJNDdefault_best_model.pth',
        input_sample_rate=16000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_sample_rate = input_sample_rate
        self.CDPAM_sample_rate = 22050  # CDPAM requirement

        encoder_layers = 16
        encoder_filters = 64
        input_size = 512
        proj_ndim = [512,256]
        ndim = [16,6]
        classif_BN = 0
        classif_act = 'no'
        proj_dp=0.1
        proj_BN=1
        classif_dp = 0.05
        
        #os.path.abspath(os.path.join(inspect.getfile(self.__init__), '..', 'weights/v%s/%s.pth'%(version,net)))
        self.model = FINnet(dev='cpu',encoder_layers=encoder_layers,encoder_filters=encoder_filters,ndim=ndim, classif_dp=classif_dp,classif_BN=classif_BN,classif_act=classif_act,input_size=input_size)
        state = torch.load(modfolder,map_location="cpu", weights_only=False)['state']
        self.model.load_state_dict(state)
        self.model.eval()

    def _prepare_audio(self, x):
        """
        1. Resample from input to 22.05kHz
        2. Scale from [-1, 1] to [-32768, 32768]
        3. Ensure mono (mean over channels if stereo)
        """
        # 1. Convert to mono if necessary (Batch, Channels, Time) -> (Batch, Time)
        if x.ndim == 3:
            x = x.mean(dim=1)
        
        # 2. Resample to 22050Hz
        if self.input_sample_rate != self.CDPAM_sample_rate:
            x = torchaudio.functional.resample(
                x, 
                self.input_sample_rate, 
                self.CDPAM_sample_rate
            )
        
        # 3. Scaling logic: match the 'np.round(inputData.astype(np.float)*32768)'
        # We use .round() and keep it as float32 for the model
        x = torch.round(x * 32768.0)
        
        return x.unsqueeze(1)

    def _get_cdpam_embeddings(self, x):
        x = self._prepare_audio(x).float()

        _, acoustics, _ = self.model.base_encoder.forward(x)
        return acoustics

    def similarity(self, x, y):
        sims = torch.zeros((x.shape[0], y.shape[0]), device=x.device)
        for i in range(x.shape[0]):
            sims[i] = -self.model.model_dist.forward(x[[i]], y).flatten()

        return sims

    def forward(self, x):
        data = torch.cat([x["anchor"], x["positive"]], dim=0)
        
        data_embeddings = self._get_cdpam_embeddings(data)

        anchor_embeddings, positive_embeddings = torch.split(
            data_embeddings, data_embeddings.size(0) // 2, dim=0
        )
        
        return self.similarity(anchor_embeddings, positive_embeddings)

    @torch.inference_mode()
    def test_step(self, x, batch_idx):
        """
        Note: This assumes a Batch x Batch comparison matrix is possible.
        CDPAM usually does 1-to-1 pair distances.
        """
        similarities = self(x)

        B = similarities.shape[0]

        labels = torch.arange(B, device=similarities.device, dtype=torch.long)
        _, predicted = torch.max(similarities, 1)
        accuracy = (predicted == labels).double().mean()

        self.log("cdpam_test_accuracy", accuracy, prog_bar=True, sync_dist=True)
        return accuracy

    def configure_optimizers(self):
        return None

if __name__ == "__main__":
    aaa = CDPAMComparisonPLWrapper()
