import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
import torchaudio

from muon import MuonWithAuxAdam

# Backends
from contrastive_model.efficientnet_backbone import EfficientNetEncoder
from contrastive_model.freq_time_separable_encoder import FreqTimeSeparableEncoder
from contrastive_model.mert import MERTEncoder

# Poolers
from contrastive_model.learned_spectral_pooling import LearnedFeaturesSpectralPooling

from contrastive_model import similarity_ops

from nnAudio.features import cqt
from feature_extraction.feature_extraction import DB201Scale

from contrastive_model import constants
from contrastive_model.complex_stuff import *

import inspect
import enum
for name, obj in inspect.getmembers(constants):
    if inspect.isclass(obj) and issubclass(obj, enum.Enum):
        torch.serialization.add_safe_globals([obj])

def make_pre_pool_projector(freq_dim, channels_dim, output_dim):
    lin = nn.Linear(freq_dim * channels_dim, output_dim, bias=False)
    nn.init.xavier_normal_(lin.weight)
    return nn.Sequential(nn.Flatten(2), lin)

class EncoderWrapper(nn.Module):
    def __init__(self,
                 embedding_dim: int = 512,
                 input_type: constants.ModelInputType = constants.ModelInputType.DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE,
                 dropout_p: float = 0.1,
                 backbone_type: str = "freqtimeseparable",
                 mert_input_sr: int = 16000,
                 mert_layer_idx: int = -1,
                 pool_type: str = "spec_pool",
                 ft_sep_init_hidden_dim: int = 8,
                 ft_sep_depth: int = 5,
                 spec_pool_out_channels: int = 80,
                 spec_pool_components: int = 8,
                 spec_pool_complex_output: bool = False,
                 spec_pool_norm: str = None,
                 spec_pool_center_padding: bool = True,
                 input_freq_bins: int = 128,
                 do_cqt: bool = False,
                 final_mlp_norm: str = "bn",
                 final_mlp: bool = True,
                 final_mlp_expansion_factor: float = None,
                 complex_mlp_bias: bool = False,
                 complex_use_silu: bool = False) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_type = input_type

        self.cqt_extractor = None
        if do_cqt:
            self.cqt_extractor = torch.nn.Sequential(
                cqt.CQT1992v2(
                    sr=16000,
                    hop_length=160,
                    n_bins=input_freq_bins,
                    fmin=16.35,
                    verbose=False
                ),
                torchaudio.transforms.AmplitudeToDB(stype='magnitude', top_db=80),
                DB201Scale(80)
            )

        in_channels = 2 if self.input_type == constants.ModelInputType.DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE else 1
        match backbone_type:
            case 'efficientnet':
                self.backbone = EfficientNetEncoder(in_channels=in_channels)
                backbone_freq_bins = input_freq_bins // 32 # efficientnet compresses in x32
                backbone_output_channels = self.backbone.model._conv_head.out_channels
            case 'freqtimeseparable' | 'axial':
                self.backbone = FreqTimeSeparableEncoder(in_channels=in_channels, init_hidden_dim=ft_sep_init_hidden_dim, depth=ft_sep_depth, dropout_p=dropout_p)
                backbone_freq_bins = input_freq_bins
                backbone_output_channels = self.backbone.output_dim
            case 'mert':
                self.backbone = MERTEncoder(input_sr=mert_input_sr, model_id="m-a-p/MERT-v1-95M", layer_idx=mert_layer_idx)
                backbone_freq_bins = 1
                backbone_output_channels = self.backbone.output_dim
            case _:
                raise NotImplementedError

        complex_mlp = False
        match pool_type:
            case 'identity':
                pooling = nn.Identity()
                pooling_out_channels = backbone_freq_bins * backbone_output_channels
            case 'avg':
                pooling = nn.AdaptiveAvgPool2d(1)
                pooling_out_channels = backbone_output_channels
            case 'max':
                pooling = nn.AdaptiveMaxPool2d(1)
                pooling_out_channels = backbone_output_channels
            case 'spec_pool' | 'fft':
                pooling = LearnedFeaturesSpectralPooling(
                    proj=make_pre_pool_projector(backbone_freq_bins, backbone_output_channels, spec_pool_out_channels),
                    out_channels=spec_pool_out_channels,
                    output_components=spec_pool_components,
                    complex_output=spec_pool_complex_output,
                    norm=spec_pool_norm,
                    center_padding=spec_pool_center_padding
                )
                pooling_out_channels = spec_pool_components * spec_pool_out_channels * (1 if spec_pool_complex_output else 1) # We now do magnitude or complex output as a 2D tensor instead of concatenated real-imag
                complex_mlp = spec_pool_complex_output
            case _:
                raise NotImplementedError

        self.pooling = nn.Sequential(
            pooling,
            CplxDropout(dropout_p) if complex_mlp else nn.Dropout(dropout_p),
            nn.Flatten(start_dim=2 if complex_mlp else 1)
        )

        if complex_mlp:
            internal_size = int(pooling_out_channels * final_mlp_expansion_factor) if final_mlp_expansion_factor is not None else self.embedding_dim

            self.mlp = nn.Sequential(
                CplxLinear(pooling_out_channels, internal_size, bias=complex_mlp_bias),
                CplxRMSNorm(internal_size),
                CplxSiLU(internal_size) if complex_use_silu else modReLU(internal_size),
                CplxLinear(internal_size, self.embedding_dim, bias=complex_mlp_bias)
            )

        else:
            internal_size = int(pooling_out_channels * final_mlp_expansion_factor) if final_mlp_expansion_factor is not None else self.embedding_dim

            match final_mlp_norm:
                case "bn":
                    norm = nn.BatchNorm1d(internal_size)
                    nn.init.ones_(norm.weight)
                    nn.init.zeros_(norm.bias)
                case "ln":
                    norm = nn.LayerNorm(internal_size)
                    nn.init.ones_(norm.weight)
                    nn.init.zeros_(norm.bias)
                case _:
                    norm = nn.Identity()

            if final_mlp:
                self.mlp = nn.Sequential(
                    nn.Linear(pooling_out_channels, internal_size),
                    norm,
                    nn.SiLU(),
                    nn.Linear(internal_size, self.embedding_dim)
                )

                nn.init.kaiming_normal_(self.mlp[0].weight, nonlinearity='relu')
                nn.init.xavier_normal_(self.mlp[3].weight)

                nn.init.zeros_(self.mlp[0].bias)
                nn.init.zeros_(self.mlp[3].bias)
            else:
                self.mlp = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cqt_extractor is not None:
            x = self.cqt_extractor(x)[:, None]
        embeddings = self.backbone(x)
        embeddings = self.pooling(embeddings)
        embeddings = self.mlp(embeddings)
        return embeddings


class ContrastiveAudioModelPLWrapper(L.LightningModule):
    def __init__(self,
                 learning_rate_muon: float = 0.02,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.0,
                 proj_weight_decay: float = 0.0,
                 embedding_dim: int = 512,
                 embedding_mode: constants.EmbeddingMode = constants.EmbeddingMode.RANDOM,
                 input_type: constants.ModelInputType = constants.ModelInputType.DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE,
                 dropout_p: float = 0.1,
                 backbone_type: str = 'freqtimeseparable',
                 mert_input_sr: int = 16000,
                 mert_layer_idx: int = -1,
                 pool_type: str = "spec_pool",
                 ft_sep_init_hidden_dim: int = 8,
                 ft_sep_depth: int = 4,
                 spec_pool_out_channels: int = 80,
                 spec_pool_components: int = 8,
                 spec_pool_complex_output: bool = False,
                 spec_pool_norm: str = None,
                 spec_pool_center_padding: bool = False,
                 input_freq_bins: int = 128,
                 comparison_method: constants.ModelComparisonMethod = constants.ModelComparisonMethod.BILINEAR_SIMILARITY,
                 label_smoothing_targ: float = 1.0,
                 do_cqt: bool = False,
                 final_mlp_norm: str = "bn",
                 final_mlp: bool = True,
                 final_mlp_expansion_factor: float = None,
                 complex_mlp_bias: bool = False,
                 complex_use_silu: bool = False,
                 mean_similarity: bool = False) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate_muon = learning_rate_muon
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.proj_weight_decay = proj_weight_decay
        self.embedding_dim = embedding_dim
        self.embedding_mode = embedding_mode
        self.comparison_method = comparison_method
        self.label_smoothing_targ = label_smoothing_targ
        self.input_type = input_type
        self.mean_similarity = mean_similarity

        self.encoder = EncoderWrapper(
            embedding_dim=self.embedding_dim,
            input_type=input_type,
            dropout_p=dropout_p,
            backbone_type=backbone_type,
            mert_input_sr=mert_input_sr,
            mert_layer_idx=mert_layer_idx,
            pool_type=pool_type,
            ft_sep_init_hidden_dim = ft_sep_init_hidden_dim,
            ft_sep_depth = ft_sep_depth,
            spec_pool_out_channels = spec_pool_out_channels,
            spec_pool_components = spec_pool_components,
            spec_pool_complex_output = spec_pool_complex_output,
            spec_pool_norm = spec_pool_norm,
            spec_pool_center_padding = spec_pool_center_padding,
            input_freq_bins = input_freq_bins,
            do_cqt = do_cqt,
            final_mlp_norm = final_mlp_norm,
            final_mlp=final_mlp,
            final_mlp_expansion_factor = final_mlp_expansion_factor,
            complex_mlp_bias = complex_mlp_bias,
            complex_use_silu = complex_use_silu
        )

        if spec_pool_complex_output:
            if self.comparison_method == constants.ModelComparisonMethod.COSINE_SIMILARITY:
                self.similarity = similarity_ops.CplxCosineSimilarity(dim=self.embedding_dim)
            elif self.comparison_method == constants.ModelComparisonMethod.BILINEAR_SIMILARITY:
                self.similarity = similarity_ops.CplxBilinearSimilarity(dim=self.embedding_dim)
            elif self.comparison_method == constants.ModelComparisonMethod.PSD_BILINEAR_SIMILARITY:
                self.similarity = similarity_ops.CplxBilinearSimilarity(dim=self.embedding_dim, psd=True)
            elif self.comparison_method == constants.ModelComparisonMethod.HERMITIAN_BILINEAR_SIMILARITY:
                self.similarity = similarity_ops.CplxBilinearSimilarity(dim=self.embedding_dim, hermitian=True)
        else:
            if self.comparison_method == constants.ModelComparisonMethod.BILINEAR_SIMILARITY:
                self.similarity = similarity_ops.BilinearSimilarity(dim=self.embedding_dim)
            elif self.comparison_method == constants.ModelComparisonMethod.COCOLA_SIMILARITY:
                self.similarity = similarity_ops.COCOLASimilarity(dim=self.embedding_dim)

    def set_embedding_mode(self, embedding_mode: constants.EmbeddingMode):
        """Sets the embedding mode for inference (for DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE models only).

        The embedding mode specifies the channel(s) to be used at inference time, a 0-mask is applied
        on the other channel(s):
        - HARMONIC: a 0-mask is applied on the percussive channel
        - PERCUSSIVE: a 0-mask is applied on the harmonic channel
        - BOTH: keeps both channels
        - RANDOM: applies one of the previous three transformations at random to each element of a batch.

        Args:
            embedding_mode (constants.EmbeddingMode): the embedding mode.
        """
        self.embedding_mode = embedding_mode

    @torch.inference_mode()
    def score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """If x and y are batches of elements, computes the COCOLA score between each element of x and y.
        Otherwise, computes the COCOLA score between x and y. 

        Args:
            x (torch.Tensor): the first batch of data of shape (B, *)
            y (torch.Tensor): the second batch of data of shape (B, *).

        Returns:
            torch.Tensor: the batch of COCOLA scores of shape (B).
        """

        data = torch.cat((x, y), dim=0)
        data_embeddings = self.encoder(data)
        x_embeddings, y_embeddings = torch.split(
            data_embeddings, data_embeddings.size(0) // 2, dim=0)

        scores = self.similarity.pairwise(x_embeddings, y_embeddings)

        return scores

    def forward(self, x):
        anchors, positives = x["anchor"], x["positive"]
        if self.input_type == constants.ModelInputType.DOUBLE_CHANNEL_HARMONIC_PERCUSSIVE:
            if self.embedding_mode == constants.EmbeddingMode.RANDOM:
                choices = torch.randint(0, 3, (anchors.shape[0],))
                anchors[choices == 0, 0, :, :] = 0
                anchors[choices == 1, 1, :, :] = 0

                positives[choices == 0, 0, :, :] = 0
                positives[choices == 1, 1, :, :] = 0
            elif self.embedding_mode == constants.EmbeddingMode.HARMONIC:
                anchors[:, 1, :, :] = 0
                positives[:, 1, :, :] = 0
            elif self.embedding_mode == constants.EmbeddingMode.PERCUSSIVE:
                anchors[:, 0, :, :] = 0
                positives[:, 0, :, :] = 0

        data = torch.cat((anchors, positives), dim=0)
        data_embeddings = self.encoder(data)
        anchor_embeddings, positive_embeddings = torch.split(
            data_embeddings, data_embeddings.size(0) // 2, dim=0)

        similarities = self.similarity(anchor_embeddings, positive_embeddings) # (B, B), each row i has sim(anchor_i, pos_j) for all j
        if self.mean_similarity:
            similarities += self.similarity(positive_embeddings, anchor_embeddings).T # (B, B) each row i has sim(pos_i, anchor_j) for all j, so transpose!
            similarities /= 2.0
        return similarities

    def training_step(self, x, batch_idx):
        similarities = self(x)
        B = similarities.shape[0]

        labels = torch.arange(
            similarities.size(0), device=similarities.device, dtype=torch.long)

        pos_weight = self.label_smoothing_targ
        neg_weight = (1.0 - pos_weight) / (B - 1)

        loss = F.cross_entropy(similarities, labels, label_smoothing=B * neg_weight)

        _, predicted = torch.max(similarities, 1)

        accuracy = (predicted == labels).double().mean()

        self.log("train_loss", loss, prog_bar=True, rank_zero_only=True)
        self.log("train_accuracy", accuracy, prog_bar=True, rank_zero_only=True)

        return loss

    def validation_step(self, x, batch_idx):
        similarities = self(x)

        labels = torch.arange(
            similarities.size(0), device=similarities.device, dtype=torch.long)

        _, predicted = torch.max(similarities, 1)
        accuracy = (predicted == labels).double().mean()

        self.log("valid_accuracy", accuracy, prog_bar=True, sync_dist=True, on_epoch=True)

    def test_step(self, x, batch_idx):
        similarities = self(x)

        labels = torch.arange(
            similarities.size(0), device=similarities.device, dtype=torch.long)

        _, predicted = torch.max(similarities, 1)
        accuracy = (predicted == labels).double().mean()

        self.log("test_accuracy", accuracy, prog_bar=True, sync_dist=True, on_epoch=True)

    def configure_optimizers(self):

        # --- 1. Define Parameter Lists ---
        muon_params = []
        adam_params = []
        proj_params = []

        # --- 2. Iterate and Assign Parameters to Groups ---
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # RULE: Assign to Adam group if it's a 1D param (bias/norm)
            # DEFAULT: All other parameters (core encoder weights) go to the Muon group
            if 'similarity' in name:
                proj_params.append(param)
            elif param.ndim < 2 or 'stem' in name or 'head' in name or 'projection' in name or "queries" in name:
                adam_params.append(param)
            else:
                muon_params.append(param)
        
        # --- 3. Create Optimizer Parameter Groups ---
        param_groups = [
            {
                'params': muon_params,
                'use_muon': True,
                'lr': self.learning_rate_muon,
                'weight_decay': self.weight_decay
            },
            {
                'params': adam_params,
                'use_muon': False,
                'lr': self.learning_rate,
                'betas': (0.9, 0.95),
                'weight_decay': self.weight_decay
            },
            {
                'params': proj_params,
                'use_muon': False,
                'lr': self.learning_rate,
                'betas': (0.9, 0.95),
                'weight_decay': self.proj_weight_decay
            },
        ]

        # --- 4. Instantiate and Return Optimizer ---
        optimizer = MuonWithAuxAdam(param_groups)

        # optimizer = torch.optim.Adam(
        #     self.parameters(),
        #     lr=self.learning_rate,
        #     weight_decay=self.weight_decay,
        #     betas=(0.9, 0.95)
        # )
        return optimizer