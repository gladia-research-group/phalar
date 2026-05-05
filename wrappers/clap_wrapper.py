import os
import torch
import torch.nn.functional as F
import laion_clap
import lightning as L
import torchaudio

class CLAPComparisonPLWrapper(L.LightningModule):
    def __init__(
        self,
        ckpt_dir=None,
        input_sample_rate = 16000,
        submodel_name="music_audioset",
        verbose=False,
        audio_load_worker=8,
        enable_fusion=False
        ):
        super().__init__()
        self.save_hyperparameters()

        assert submodel_name in ["630k-audioset", "630k", "music_audioset", "music_speech", "music_speech_audioset"]
        self.submodel_name = submodel_name
        self.input_sample_rate = input_sample_rate
        self.CLAP_sample_rate = 48000  # CLAP model sample rate
        self.verbose = verbose
        self.audio_load_worker = audio_load_worker
        self.enable_fusion = enable_fusion
        if ckpt_dir is not None:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.hub.set_dir(ckpt_dir)
            self.ckpt_dir = ckpt_dir
        else:
            # by default `ckpt_dir` is `torch.hub.get_dir()`
            self.ckpt_dir = torch.hub.get_dir()
        self.__get_model()

        for param in self.clap.parameters():
            param.requires_grad = False

    def __get_model(self):
        """
        Get ckpt and set model for the specified model_name
        """
        # choose the right checkpoint and model
        if self.submodel_name == "630k-audioset":
            if self.enable_fusion:
                download_name = "630k-audioset-fusion-best.pt"
            else:
                download_name = "630k-audioset-best.pt"
        elif self.submodel_name == "630k":
            if self.enable_fusion:
                download_name = "630k-fusion-best.pt"
            else:
                download_name = "630k-best.pt"
        elif self.submodel_name == "music_audioset":
            download_name = "music_audioset_epoch_15_esc_90.14.pt"
        elif self.submodel_name == "music_speech":
            download_name = "music_speech_epoch_15_esc_89.25.pt"
        elif self.submodel_name == "music_speech_audioset":
            download_name = "music_speech_audioset_epoch_15_esc_89.98.pt"

        model_path = os.path.join(self.ckpt_dir, download_name)

        # download checkpoint
        if not (os.path.exists(model_path)):
            if self.verbose:
                print("[CLAP Score] Downloading {}...".format(model_path))
            torch.hub.download_url_to_file(
                url=f"https://huggingface.co/lukewys/laion_clap/resolve/main/{download_name}",
                dst=model_path
            )

        # init model and load checkpoint
        if self.submodel_name in ["630k-audioset", "630k"]:
            self.clap = laion_clap.CLAP_Module(enable_fusion=self.enable_fusion)
        elif self.submodel_name in ["music_audioset", "music_speech", "music_speech_audioset"]:
            self.clap = laion_clap.CLAP_Module(enable_fusion=self.enable_fusion,
                                                amodel='HTSAT-base')
        self.clap.load_ckpt(model_path)
        self.clap.eval()

    def _get_clap_embeddings(self, x):
        """Processes the H-P channels to match your model's logic"""
        if x.dim() == 3: # (B, C, T)
            x = x.mean(dim=1) 
        
        # Resample to 48kHz
        x_resampled = torchaudio.functional.resample(x, self.input_sample_rate, self.CLAP_sample_rate)
        
        # CLAP internal preprocessing (quantization/scaling)
        # Note: 'get_audio_embedding_from_data' expects (N, T)
        return self.clap.get_audio_embedding_from_data(x=x_resampled, use_tensor=True)

    def similarity(self, x,y):
        # CLAP embeddings are already L2 normalized
        # x = F.normalize(x, p=2, dim=1)
        # y = F.normalize(y, p=2, dim=1)
        
        similarities = torch.matmul(x, y.T)
        return similarities

    def forward(self, x):
        anchors, positives = x["anchor"], x["positive"]
        
        # Concatenate for batch efficiency
        data = torch.cat((anchors, positives), dim=0)
        
        # Get embeddings
        data_embeddings = self._get_clap_embeddings(data)
        
        anchor_embeddings, positive_embeddings = torch.split(
            data_embeddings, data_embeddings.size(0) // 2, dim=0
        )

        return self.similarity(anchor_embeddings, positive_embeddings)

    @torch.inference_mode()
    def test_step(self, x, batch_idx):
        similarities = self(x)
        B = similarities.shape[0]

        # Your random negative picking logic for exact parity
        labels = torch.arange(B, device=similarities.device, dtype=torch.long)
        _, predicted = torch.max(similarities, 1)
        accuracy = (predicted == labels).double().mean()

        self.log("clap_test_accuracy", accuracy, prog_bar=True, sync_dist=True)
        return accuracy

    def configure_optimizers(self):
        return None # Test only