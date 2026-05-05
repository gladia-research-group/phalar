import random
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from typing import Dict, Literal
from pathlib import Path
import logging
from feature_extraction.feature_extraction import waveform_audio_augmentations, post_transform_augmentations
import numpy as np
import librosa
import math

from data.utils import right_pad, mix_down, mix_stems
from contrastive_model import constants

class OnlineDataset(Dataset):
    """
    General Dataset with random overlap between chunks for contrastive learning.
    """

    def __init__(
            self,
            chunk_duration=5,
            target_sample_rate=16000,
            generate_submixtures=True,
            preprocess_transform=None,
            runtime_transform=None,
            augmentations=dict(),
            feature_extractor_type: constants.ModelFeatureExtractorType = constants.ModelFeatureExtractorType.STFT_SPECTROGRAM,
            mono=True,
            ) -> None:

        self.chunk_duration = chunk_duration
        self.target_sample_rate = target_sample_rate
        self.generate_submixtures = generate_submixtures
        self.preprocess_transform = preprocess_transform
        self.runtime_transform = runtime_transform
        self.augmentations = augmentations
        self.feature_extractor_type = feature_extractor_type
        self.mono = mono

        self._check_and_fill_augments()

        self.track_index = []

    def _check_and_fill_augments(self):
        """
        Ensure every expected augmentation entry exists in the config.
        If missing, create it with p = -1.
        """
        # Define your expected structure
        expected = ["pitch_shift", "time_stretch", "random_gain",
                    "white_noise", "pink_noise", "brown_noise",
                    "band_limited_noise", "transient_noise", "freq_shift", "gaussian_noise"]

        # Walk the structure and fill missing nodes
        for entry in expected:
            if entry not in self.augmentations:
                self.augmentations[entry] = {"p":-1}

    def __len__(self) -> int:
        return -1

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        max_retries = 5
        for _ in range(max_retries):
            track_info = random.choice(self.track_index)
            try:
                return self._get_item_from_track(track_info)
            except Exception as e:
                print(f"Error loading track: {e}")
                if _ == max_retries - 1:
                    raise e

    def _get_item_from_track(self, track_info):

        # Sample joint augmentations
        semitones = None
        rate = None
        gain = None
        bins_shift = None

        if np.random.uniform() < self.augmentations["pitch_shift"]["p"]:
            semitones = np.random.uniform(*self.augmentations["pitch_shift"]["semitones"])
            if abs(semitones) < 1e-4:
                semitones = None

        if np.random.uniform() < self.augmentations["time_stretch"]["p"]:
            rate = np.random.uniform(*self.augmentations["time_stretch"]["rate"])
            if abs(rate - 1) < 1e-4:
                rate = None

        if np.random.uniform() < self.augmentations["random_gain"]["p"]:
            gain = np.random.uniform(*self.augmentations["random_gain"]["db"])
            if abs(gain) < 1e-4:
                gain = None

        if np.random.uniform() < self.augmentations["freq_shift"]["p"]:
            bins_shift = np.random.randint(*self.augmentations["freq_shift"]["bins"])
            if abs(bins_shift) < 1:
                bins_shift = None

        stems_paths = track_info['stems_paths']
        #print(stems_paths)
        sample_rate = track_info['sample_rate']
        num_frames = track_info['num_frames']
        chunk_num_frames = int(self.chunk_duration * sample_rate)

        if rate:
            chunk_num_frames = int(chunk_num_frames * rate)
            rate = chunk_num_frames / int(self.chunk_duration * sample_rate)

        # Randomly select a starting frame
        max_start_frame = num_frames - chunk_num_frames
        if max_start_frame <= 0:
            frame_offset = 0
        else:
            frame_offset = random.randint(0, max_start_frame)

        stems_idxs = list(range(len(stems_paths)))

        if self.generate_submixtures and len(stems_idxs) > 1:
            anchor_mix_size = random.randint(1, len(stems_idxs) - 1)
            anchor_mix_idxs = random.sample(stems_idxs, anchor_mix_size)

            positive_mix_size = random.randint(1, len(stems_idxs) - len(anchor_mix_idxs))
            positive_mix_idxs = random.sample(
                [idx for idx in stems_idxs if idx not in anchor_mix_idxs], positive_mix_size)
        else:
            anchor_mix_idxs = random.sample(stems_idxs, 1)
            positive_mix_idxs = random.sample(
                [idx for idx in stems_idxs if idx not in anchor_mix_idxs], 1)

        stems = dict()
        for stem_idx in set(anchor_mix_idxs + positive_mix_idxs):
            stem_path = stems_paths[stem_idx]
            try:
                waveform, sr = torchaudio.load(
                    str(stem_path),
                    frame_offset=frame_offset,
                    num_frames=chunk_num_frames,
                    backend="soundfile"
                )
            except Exception as e:
                # If a stem fails to load, raise an error to trigger the retry logic
                raise RuntimeError(f"Error loading {stem_path}, sr {sample_rate}, starting frame {frame_offset}, num_frames {chunk_num_frames}, max_starting_frame {max_start_frame}: {e}")
            
            if sr != self.target_sample_rate:
                try:
                    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.target_sample_rate)
                except Exception as e:
                    raise RuntimeError(f"Error resampling {stem_path}, sr {sr}, starting frame {frame_offset}, num_frames {chunk_num_frames}, max_starting_frame {max_start_frame}: {e}")
            
            # Mix down to mono
            if self.mono:
                waveform = mix_down(waveform)

            stems[stem_idx] = waveform

        resampled_num_frames = int(self.target_sample_rate * self.chunk_duration)

        anchor = mix_stems(
            [right_pad(stems[j], resampled_num_frames) for j in anchor_mix_idxs])
        positive = mix_stems(
            [right_pad(stems[j], resampled_num_frames) for j in positive_mix_idxs])

        if self.preprocess_transform:
            anchor = self.preprocess_transform(anchor)
            positive = self.preprocess_transform(positive)

        anchor = anchor.cpu().numpy()
        positive = positive.cpu().numpy()

        anchor = waveform_audio_augmentations(anchor, semitones=semitones, rate=rate, gain=gain, augmentations=self.augmentations)
        positive = waveform_audio_augmentations(positive, semitones=semitones, rate=rate, gain=gain, augmentations=self.augmentations)

        item = {"anchor": torch.from_numpy(anchor).float(), "positive": torch.from_numpy(positive).float()}

        item = self.runtime_transform(item)

        if self.feature_extractor_type != constants.ModelFeatureExtractorType.RAW_WAVEFORM:
            item["anchor"] = post_transform_augmentations(item["anchor"], bins_shift=bins_shift, augmentations=self.augmentations)
            item["positive"] = post_transform_augmentations(item["positive"], bins_shift=bins_shift, augmentations=self.augmentations)

        return item