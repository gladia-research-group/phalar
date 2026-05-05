"""Audio preprocessing utility classes."""

from typing import Union

import numpy as np
import torch
import torchaudio.transforms as T
from torch import nn
import librosa

from contrastive_model import constants
from nnAudio.features import cqt


def freq_shift_stft(S, shift_bins):
    """Shift STFT upward along the frequency axis (zero-fill bottom)."""
    if shift_bins == 0:
        return S
    S_shifted = torch.zeros_like(S, dtype=S.dtype)
    if shift_bins > 0:
        S_shifted[shift_bins:, :] = S[:-shift_bins, :]
    else:
        S_shifted[:shift_bins, :] = S[shift_bins:, :]
    return S_shifted

def gain_change(waveform, db=0):
    gain = 10 ** (db / 20)
    return waveform * gain

def white_noise(waveform, snr_db=(14, 20)):
    snr = np.random.uniform(*snr_db)
    rms_signal = (waveform ** 2).mean() ** .5
    rms_noise = rms_signal / (10 ** (snr / 20))
    noise = np.random.normal(0, rms_noise, waveform.shape)
    return waveform + noise

def pink_noise(waveform, snr_db=(14, 20)):
    length = waveform.shape[-1]
    nrows = 16
    array = np.random.randn(nrows, length).cumsum(axis=1)
    weights = 2 ** np.arange(nrows)
    pink = (array / weights[:, None]).sum(axis=0)
    pink = pink / np.std(pink)
    snr = np.random.uniform(*snr_db)
    rms_signal = (waveform ** 2).mean() ** .5
    rms_noise = rms_signal / (10 ** (snr / 20))
    return waveform + pink * rms_noise

def brown_noise(waveform, snr_db=(12, 18)):
    length = waveform.shape[-1]
    w = np.random.randn(1, length)
    b = w.cumsum(0)
    b = b / b.std()
    snr = np.random.uniform(*snr_db)
    rms_signal = (waveform ** 2).mean() ** .5
    rms_noise = rms_signal / (10 ** (snr / 20))
    return waveform + b * rms_noise

from scipy.signal import butter, lfilter
def butter_bandpass(lowcut, highcut, fs=16000, order=3):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs=16000, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def band_limited_noise(waveform, sr=16000, snr_db=(12, 18), band=(200, 8000)):
    # pick a random low/high cutoff
    low = np.random.randint(band[0], band[1] // 2)
    high = np.random.randint(low + 100, band[1])  # ensure high > low
    noise = np.random.randn(*waveform.shape)
    noise = butter_bandpass_filter(noise, low, high, fs=16000, order=3)
    
    rms_signal = (waveform ** 2).mean() ** .5
    snr = np.random.uniform(*snr_db)
    rms_noise = (noise ** 2).mean() ** .5
    if rms_noise > 0:
        noise = noise * (rms_signal / (10 ** (snr / 20)) / rms_noise)
    return waveform + noise

def noise_burst(waveform, burst_length=(10, 21), snr_db=(6, 12)):
    burst_length = np.random.randint(*burst_length)
    snr = np.random.uniform(*snr_db)
    start = np.random.randint(0, waveform.shape[1]-burst_length)
    burst = np.random.randn(1, burst_length)
    # scale burst to desired SNR
    rms_signal = (waveform ** 2).mean() ** .5
    rms_noise = (burst ** 2).mean() ** .5
    burst = burst * (rms_signal / (10**(snr/20)) / (rms_noise + 1e-9))
    waveform[:, start:start+burst_length] += burst
    return waveform

def gaussian_noise(spec, std_range=(0.0, 0.02)):
    std = np.random.uniform(*std_range)
    noise = torch.randn(spec.shape) * std
    return spec + noise

def waveform_audio_augmentations(y, augmentations, semitones=None, rate=None, gain=None):
    if rate:
        y = librosa.effects.time_stretch(y, rate=rate)

    if semitones:
        y = librosa.effects.pitch_shift(y, sr=16000, n_steps=semitones)

    if gain:
        y = gain_change(y, db=gain)

    if np.random.uniform() < augmentations["white_noise"]["p"]:
        y = white_noise(y, snr_db=augmentations["white_noise"]["snr_db"])

    if np.random.uniform() < augmentations["pink_noise"]["p"]:
        y = pink_noise(y, snr_db=augmentations["pink_noise"]["snr_db"])

    if np.random.uniform() < augmentations["brown_noise"]["p"]:
        y = brown_noise(y, snr_db=augmentations["brown_noise"]["snr_db"])

    if np.random.uniform() < augmentations["band_limited_noise"]["p"]:
        y = band_limited_noise(y, sr=16000, snr_db=augmentations["band_limited_noise"]["snr_db"], band=augmentations["band_limited_noise"]["band"])

    if np.random.uniform() < augmentations["transient_noise"]["p"]:
        for _ in range(np.random.randint(*augmentations["transient_noise"]["num_bursts"])):
            y = noise_burst(y, burst_length=augmentations["transient_noise"]["burst_length"], snr_db=augmentations["transient_noise"]["snr_db"])
    return y

def post_transform_augmentations(y, augmentations, bins_shift=None):
    if bins_shift:
        y = freq_shift_stft(y, bins_shift)

    if np.random.uniform() < augmentations["gaussian_noise"]["p"]:
        y = gaussian_noise(y, std_range=augmentations["gaussian_noise"]["std"])

    return y

class HPSS(nn.Module):
    def __init__(self,
                 sample_rate: int = 16000,
                 f_min: float = 60.0,
                 f_max: float = 7800.0,
                 n_mels: int = 64) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels

    def forward(self, x: torch.Tensor):
        """Extract HPSS feature tensor(s) from input audio tensor(s).

        Args:
            x (torch.Tensor): The audio tensor(s) of shape (B, 1, S) or (1, S).

        Returns:
            torch.Tensor: The HPSS features tensor(s) of shape (B, 2, H, W) or (2, H, W).
        """

        x = x.cpu().numpy()


        harmonic_stft, percussive_stft = librosa.decompose.hpss(x)


        mel_harmonic = librosa.feature.melspectrogram(S=np.abs(harmonic_stft)**2,
                                                        sr=self.sample_rate,
                                                        fmin=self.f_min,
                                                        fmax=self.f_max,
                                                        n_mels=self.n_mels)


        mel_percussive = librosa.feature.melspectrogram(S=np.abs(percussive_stft)**2,
                                                        sr=self.sample_rate,
                                                        fmin=self.f_min,
                                                        fmax=self.f_max,
                                                        n_mels=self.n_mels)

        mel_db_harmonic = librosa.power_to_db(mel_harmonic, ref=np.max)

        mel_db_percussive = librosa.power_to_db(mel_percussive, ref=np.max)

        hp_mel_db = np.concatenate(
            (mel_db_harmonic, mel_db_percussive), axis=0)

        hp_mel_db = torch.from_numpy(hp_mel_db)

        return hp_mel_db


class ComplexToPower(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        """Converts complex tensor to power spectrogram.

        Args:
            x (torch.Tensor): Complex tensor of shape (..., F, T).

        Returns:
            torch.Tensor: Power spectrogram tensor of shape (..., F, T).
        """
        return (x.abs() ** 2).float()

class DB201Scale(nn.Module):
    def __init__(self, top_db: float = 80.0):
        super().__init__()
        self.top_db = top_db

    def forward(self, x):
        return (x + self.top_db) / self.top_db


class FeatureExtractor(nn.Module):
    def __init__(self,
                 feature_extractor_type: constants.ModelFeatureExtractorType = constants.ModelFeatureExtractorType.HPSS,
                 sample_rate: int = 16000,
                 n_fft: int = 1024,
                 win_length: int = 400,
                 hop_length: int = 160,
                 f_min: float = 60.0,
                 f_max: float = 7800.0,
                 n_mels: int = 128,
                 n_cqt_bins: int = 96,
                 top_db: float = 80.0) -> None:
        super().__init__()
        self.feature_extractor_type = feature_extractor_type
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        self.n_cqt_bins = n_cqt_bins
        self.top_db = top_db

        if self.feature_extractor_type == constants.ModelFeatureExtractorType.HPSS:
            self.feature_extractor = nn.Sequential(
                T.Spectrogram(
                    n_fft=self.n_fft,
                    win_length=self.win_length,
                    hop_length=self.hop_length,
                    power=None
                ),
                HPSS(
                    sample_rate=self.sample_rate,
                    f_min=self.f_min,
                    f_max=self.f_max,
                    n_mels=self.n_mels // 2
                )
            )
        elif self.feature_extractor_type == constants.ModelFeatureExtractorType.MEL_SPECTROGRAM:
            self.feature_extractor = torch.nn.Sequential(
                T.Spectrogram(
                    n_fft=self.n_fft,
                    win_length=self.win_length,
                    hop_length=self.hop_length
                ),
                T.MelScale(
                    sample_rate=self.sample_rate,
                    n_stft=self.n_fft // 2 + 1,
                    f_min=self.f_min,
                    f_max=self.f_max,
                    n_mels=self.n_mels
                ),
                T.AmplitudeToDB(stype='power', top_db=self.top_db),
                DB201Scale(self.top_db)
            )
        elif self.feature_extractor_type == constants.ModelFeatureExtractorType.STFT_SPECTROGRAM:
            self.feature_extractor = torch.nn.Sequential(
                T.Spectrogram(
                    n_fft=self.n_fft,
                    win_length=self.win_length,
                    hop_length=self.hop_length
                ),
                T.AmplitudeToDB(stype='power', top_db=self.top_db),
                DB201Scale(self.top_db)
            )
        elif self.feature_extractor_type == constants.ModelFeatureExtractorType.CQT_SPECTROGRAM:
            self.feature_extractor = torch.nn.Sequential(
                cqt.CQT1992v2(
                    sr=self.sample_rate,
                    hop_length=self.hop_length,
                    n_bins=self.n_cqt_bins,
                    fmin=16.35,
                    verbose=False
                ),
                T.AmplitudeToDB(stype='magnitude', top_db=self.top_db),
                DB201Scale(self.top_db)
            )
        elif self.feature_extractor_type == constants.ModelFeatureExtractorType.RAW_WAVEFORM:
            self.feature_extractor = torch.nn.Identity()

    def forward(self, x: Union[dict, torch.Tensor]):
        """Performs feature extraction.

        Args:
            x (Union[dict, torch.Tensor]): the waveform or anchor positive dictionary on which feature extraction is applied.

        Returns:
            Union[dict, torch.Tensor]: features tensor or dictionary.
        """
        if isinstance(x, dict):
            a, p = x['anchor'], x['positive']
            x['anchor'] = self.feature_extractor(a)
            x['positive'] = self.feature_extractor(p)
            return x
        else:
            return self.feature_extractor(x)

    def to_dict(self):
        """Used for serialization."""
        return {
            "feature_extractor_type": self.feature_extractor_type.value,
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "win_length": self.win_length,
            "hop_length": self.hop_length,
            "f_min": self.f_min,
            "f_max": self.f_max,
            "n_mels": self.n_mels,
            "n_cqt_bins": self.n_cqt_bins,
            "top_db": self.top_db
        }
