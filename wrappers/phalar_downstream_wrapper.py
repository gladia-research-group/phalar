from __future__ import annotations
import os
import sys

# Adds the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List, Optional, Tuple

import torch
import torchaudio
import torchaudio.functional as F

import librosa
import numpy as np

from synth_utils import *
from contrastive_model.contrastive_model import ContrastiveAudioModelPLWrapper

def get_normalization_factor(layer):
    # 1. Take weight from CplxLinear layer
    w_raw = layer.W.weight.detach()
    out_dim = w_raw.shape[0] // 2
    
    # 2. Take real A and imaginary B parts
    A = w_raw[:out_dim, :] 
    B = w_raw[out_dim:, :]
    
    # 3. Reconstruct the Complex Weight Matrix W for pytorch's sake
    W = A + 1j * B
    
    # 4. Compute the Hermitian part
    W_eff = 0.5 * (W + W.conj().T)
    
    # 5. Compute Eigenvalue
    eigvals = torch.linalg.eigvalsh(W_eff)
    
    # 6. Get max absolute eigenvalue
    rho = torch.max(torch.abs(eigvals))
    
    return rho.item()


class PHALARWrapper:
    """
    Improved PHALAR wrapper for beat tracking + chord recognition.

    Key improvements vs the original snippet:
    - Beat-sync pooling no longer "breaks" leaving zero-columns that pollute Viterbi.
    - Optional time alignment for pooling using the *center* of the analysis window.
    - Emissions are computed with log_softmax (numerically stable).
    - Viterbi can use an optional harmonic-aware transition matrix based on chord note overlap.
    """

    SAMPLE_RATE = 16000
    HOP_LENGTH = 160
    SAMPLES_PER_SECOND = SAMPLE_RATE / HOP_LENGTH  # 100.0 frames/s at hop=160
    DOWNSIZE_FACTOR = 32

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        first_beat_start: float = 0.02,
        beat_tracking_chunks_duration: float = 5.0,
        bpm_range: Tuple[int, int] = (30, 240),
        fluidsynth_soundfont: Optional[str] = None,
    ):
        # NOTE: these symbols must exist in your codebase.
        # If you keep them in different modules, update the imports accordingly.
        #
        # from your_pkg import ContrastiveAudioModelPLWrapper, FluidSynthTemplateGenerator, get_normalization_factor
        try:
            ContrastiveAudioModelPLWrapper  # type: ignore[name-defined]  # noqa: B018
            FluidSynthTemplateGenerator  # type: ignore[name-defined]  # noqa: B018
            get_normalization_factor  # type: ignore[name-defined]  # noqa: B018
        except NameError as e:  # pragma: no cover
            raise ImportError(
                "Missing project-specific symbols. Ensure the following are importable in this module: "
                "`ContrastiveAudioModelPLWrapper`, `FluidSynthTemplateGenerator`, `get_normalization_factor`."
            ) from e

        checkpoint = torch.load(model_path, map_location="cpu")

        self.fs_gen = None
        if fluidsynth_soundfont is not None:
            self.fs_gen = FluidSynthTemplateGenerator(
                fluidsynth_soundfont, target_fs=self.SAMPLE_RATE
            )

        self.device = device

        model = ContrastiveAudioModelPLWrapper(**checkpoint["hyper_parameters"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model = model.to(self.device)
        self.model.eval()
        self.model.similarity.eval()

        # Model-specific settings (kept from your snippet)
        self.model.encoder.pooling[0].center_padding = False
        self.model.similarity.hermitian = True  # avoid redundant computations

        # normalization factor (kept for parity with your snippet)
        self.max_lambda = get_normalization_factor(self.model.similarity.W)

        self.beat_sim = lambda x, y: self.model.similarity(x, y)
        self.chord_sim = lambda x, y: self.model.similarity(x, y)
        self.chord_sim_pairwise = lambda x, y: self.model.similarity.pairwise(x, y)

        self.first_beat_start = float(first_beat_start)
        self.beat_tracking_chunks_duration = float(beat_tracking_chunks_duration)
        self.bpm_range = bpm_range

        lo, hi = int(bpm_range[0]), int(bpm_range[1])
        self.bpms = list(range(lo, hi + 1))
        self.bpm_embeddings = self._get_bpm_embeddings()

    @torch.inference_mode()
    def _get_bpm_embeddings(self) -> torch.Tensor:
        if self.fs_gen is None:
            raise ValueError(
                "FluidSynth generator is not initialized; pass `fluidsynth_soundfont=`."
            )

        metronome_waveforms: List[torch.Tensor] = []
        for bpm in self.bpms:
            met = self.fs_gen.generate_metronome(
                bpm,
                duration_seconds=self.beat_tracking_chunks_duration,
                offset_s=self.first_beat_start,
            )
            metronome_waveforms.append(torch.from_numpy(met).float())

        bpm_audios = torch.stack(metronome_waveforms, dim=0).to(self.device)
        bpm_embeddings = self.model.encoder(bpm_audios)
        return bpm_embeddings

    @torch.inference_mode()
    def _get_audio_embeddings(
        self,
        audio_path: str,
        chunks_len: float = 5.0,
        resolution: float = 0.02,
        batch_size: int = 1024,
        padding_seconds: float = 0.0,
    ):
        embeddings_size = int((self.SAMPLES_PER_SECOND * chunks_len) // self.DOWNSIZE_FACTOR)
        size = embeddings_size * self.DOWNSIZE_FACTOR

        # Print beats over spectogram
        if isinstance(audio_path, str):
            waveform, sr = torchaudio.load(audio_path, normalize=True)
            waveform = F.resample(waveform, sr, self.SAMPLE_RATE)
            waveform = waveform.squeeze()
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=0)  # mix down to mono
        else:
            waveform = audio_path

        if padding_seconds > 0.0:
            pad_len = int(padding_seconds * self.SAMPLES_PER_SECOND)
            left_pad = torch.zeros(pad_len, dtype=waveform.dtype, device=waveform.device)
            right_pad = torch.zeros(pad_len, dtype=waveform.dtype, device=waveform.device)
            waveform = torch.cat([left_pad, waveform, right_pad], dim=0)

        full_audio = waveform[None]

        stride = int(resolution * self.SAMPLES_PER_SECOND)

        full_audio = full_audio.to(self.device)
        full_audio_cqt = self.model.encoder.cqt_extractor(full_audio)[:, None].cpu() # 1, 1, F, T

        del full_audio
        torch.cuda.empty_cache()

        all_embeddings = []
        with torch.autocast(enabled=True, device_type=self.device, dtype=torch.float32 if self.device.startswith('cuda') else torch.bfloat16):
            for start_idx in range(0, full_audio_cqt.shape[-1]-size+1, stride*batch_size):
                x = full_audio_cqt[..., start_idx:start_idx+stride*(batch_size-1)+size].to(self.device) # 1, 1, F, T'

                x = x.unfold(dimension=-1, size=size, step=stride).permute(0, 3, 1, 2, 4)  # 1, num_crops, 1, F, size
                x = x.reshape(-1, 1, x.shape[-2], x.shape[-1])  # (1 * num_crops), 1, F, size

                x = self.model.encoder.backbone(x)
                x = self.model.encoder.pooling(x)
                x = self.model.encoder.mlp(x)
                all_embeddings.append(x.cpu())
                
        del full_audio_cqt
        torch.cuda.empty_cache()

        embeddings = torch.cat(all_embeddings, dim=0).to(device=self.device, dtype=torch.float32)  # (E, C)

        return embeddings


    @torch.inference_mode()
    def track_beats(self, audio_path: str, resolution: float = 0.02):
        audio_embeddings = self._get_audio_embeddings(
            audio_path,
            chunks_len=self.beat_tracking_chunks_duration,
            resolution=resolution,
            batch_size=1024
        )

        raw_scores = self.beat_sim(self.bpm_embeddings, audio_embeddings) # (E, B)

        # Show heatmap of signals
        # plt.imshow(raw_scores.cpu().numpy(), aspect='auto', origin='lower', cmap='viridis', extent=[0, raw_scores.shape[1]*resolution, self.bpm_range[0], self.bpm_range[1]], interpolation='nearest')
        # plt.colorbar(label='Similarity scores')
        # plt.xlim(0, 30.0)   # Show only first 30 seconds for clarity
        # plt.xlabel('Time (s)')
        # plt.ylabel('BPMs')
        # plt.savefig("beat_similarity_heatmap.png", dpi=300, bbox_inches='tight')
        # plt.close()

        # Compute derivative along time axis
        derivative_filter = torch.tensor(
            [[-1.0, 0, 1.0]],
            device=self.device
        )
        derivative_scores = torch.nn.functional.conv2d(raw_scores[None, None, :, :], derivative_filter[None, None, :, :], padding=(0,1)).squeeze()

        # Suppress noise in derivative scores
        mask = ((derivative_scores - derivative_scores.mean()) > (3.0*derivative_scores.std()))
        # erode
        mask = mask.float()
        sum_filter = torch.ones((1,1,3,3), device=self.device)
        count = torch.nn.functional.conv2d(mask[None, None, :, :], sum_filter, padding=1).squeeze()
        mask = (count >= 3).float()
        # dilate
        count = torch.nn.functional.conv2d(mask[None, None, :, :], sum_filter, padding=1).squeeze()
        mask = (count >= 1).float()
        derivative_scores = torch.clamp(derivative_scores * mask, min=0.0)

        onset_signal = derivative_scores.sum(dim=0).cpu().numpy()
   
        stride = int(resolution * self.SAMPLES_PER_SECOND)

        tempo, beat_times = librosa.beat.beat_track(
            onset_envelope=onset_signal,
            sr=self.SAMPLE_RATE,
            # start_bpm=tempo_estimate,
            hop_length=stride * self.HOP_LENGTH,
            units='time',
            # tightness=100.0
        )

        beat_times = beat_times + self.first_beat_start
        tempo_val = float(tempo.item() if isinstance(tempo, np.ndarray) else tempo)
        return beat_times, tempo_val

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # from beat_this.inference import File2Beats


    model_path = "PHALAR_best.ckpt"
    fluidsynth_soundfont = "GeneralUser-GS.sf2"
    device = "cuda:0"
    audio_path = "I Want to Live (Classical Version).flac"

    phalar_wrapper = PHALARWrapper(
        model_path=model_path,
        device=device,
        fluidsynth_soundfont=fluidsynth_soundfont
    )

    beat_times, tempo_estimate = phalar_wrapper.track_beats(
        audio_path,
        resolution=0.02
    )

    # Vizualize Beat Tracking over spectrogram
    plt.figure(figsize=(24, 6))
    waveform, sr = torchaudio.load(audio_path, normalize=True)
    waveform = F.resample(waveform, sr, phalar_wrapper.SAMPLE_RATE)
    waveform = waveform.squeeze().cpu().numpy()
    MAX = 30.0
    waveform = waveform[..., :int(phalar_wrapper.SAMPLE_RATE * MAX)]
    # mono
    waveform = waveform.mean(axis=0) if waveform.ndim > 1 else waveform

    # file2beats = File2Beats(checkpoint_path="final0", device="cuda", dbn=False)
    # beats, downbeats = file2beats(str(audio_path))

    D = librosa.amplitude_to_db(np.abs(librosa.stft(waveform, n_fft=1024, hop_length=phalar_wrapper.HOP_LENGTH)), ref=np.max)
    librosa.display.specshow(D, sr=phalar_wrapper.SAMPLE_RATE, hop_length=phalar_wrapper.HOP_LENGTH, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    for bt in beat_times:
        if bt > MAX:
            continue
        plt.axvline(x=bt, color='cyan', alpha=0.5, linewidth=2, label='PHALAR Beat')
    # for bt in beats:
    #     if bt > MAX:
    #         continue
    #     plt.axvline(x=bt, color='yellow', alpha=0.5, linewidth=2, label='Beat This!')
    plt.title('Beat Tracking Overlayed on Spectrogram')
    plt.savefig("beat_tracking.png", dpi=300)
    plt.close()
