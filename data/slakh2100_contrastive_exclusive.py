"""Slakh2100 Contrastive Torch Dataset (Online Version)."""

from pathlib import Path
import random
import logging

from torchvision.datasets.utils import download_and_extract_archive
import torchaudio
from tqdm import tqdm
import torch

from data import OnlineDataset
from typing import Dict

from contrastive_model import constants

random.seed(14703)

class Slakh2100ContrastiveExclusive(OnlineDataset):
    """
    Slakh2100 Dataset (adapted for contrastive learning, online loading):
    http://www.slakh.com
    """

    VERSION = "1.0.0"
    URL = "https://zenodo.org/records/7708270/files/slakh2100_redux_16k.tar.gz"
    SAMPLE_RATE = 16000
    ORIGINAL_DIR_NAME = "original"

    def __init__(
            self,
            root_dir="~/slakh2100_contrastive",
            download=True,
            split="train",
            chunk_duration=5,
            target_sample_rate=16000,
            generate_submixtures=True,
            device="cpu",
            preprocess_transform=None,
            runtime_transform=None,
            samples_per_epoch=10000,
            augmentations=dict(),
            feature_extractor_type: constants.ModelFeatureExtractorType = constants.ModelFeatureExtractorType.STFT_SPECTROGRAM,
            mono=True) -> None:

        super().__init__(
            chunk_duration=chunk_duration,
            target_sample_rate=target_sample_rate,
            generate_submixtures=generate_submixtures,
            preprocess_transform=preprocess_transform,
            runtime_transform=runtime_transform,
            augmentations=augmentations,
            feature_extractor_type=feature_extractor_type,
            mono=mono
        )

        self.root_dir = Path(root_dir).expanduser()
        self.download = download
        self.split = split
        self.samples_per_epoch = samples_per_epoch
        self.device = device

        if self.split not in ["train", "test", "validation"]:
            raise ValueError("`split` must be one of ['train', 'test', 'validation'].")

        # Scarica il dataset se non presente
        if self.download and not self._is_downloaded_and_extracted():
            self._download_and_extract()

        if not self._is_downloaded_and_extracted():
            raise RuntimeError(
                f"Dataset split {self.split} not found. Please set `download=True` or place the data properly.")
        logging.info(
            f"Found original dataset split {self.split} at {(self.root_dir / self.ORIGINAL_DIR_NAME / 'slakh2100_redux_16k' / self.split)}.")

        # Costruisce l'indice dei brani (track e stems)
        self._build_index()

    def _is_downloaded_and_extracted(self) -> bool:
        split_dir = (self.root_dir / self.ORIGINAL_DIR_NAME / "slakh2100_redux_16k" /
                     self.split)
        return split_dir.exists() and any(split_dir.iterdir())

    def _download_and_extract(self) -> None:
        download_and_extract_archive(
            self.URL, self.root_dir / self.ORIGINAL_DIR_NAME, remove_finished=True)

    def _build_index(self):
        original_dir = (self.root_dir / self.ORIGINAL_DIR_NAME /
                        "slakh2100_redux_16k" / self.split)
        tracks = list(original_dir.glob("*/"))
        if not tracks:
            raise RuntimeError(f"No tracks found in split {self.split}.")

        self.track_index = []
        for track in tqdm(tracks, desc="Building track index"):
            try:
                # Carica tutti gli stems .flac nella cartella stems
                stems_paths = list(track.glob("stems/S*.wav"))
                if not stems_paths:
                    continue

                lengths = []
                for stem in stems_paths:
                    info = torchaudio.info(stem, backend="soundfile")
                    lengths.append(info.num_frames)

                # Assume che tutti gli stems abbiano la stessa durata
                info = torchaudio.info(str(stems_paths[0]))
                num_frames = min(lengths)
                sample_rate = info.sample_rate

                self.track_index.append({
                    'track_name': track.name,
                    'stems_paths': stems_paths,
                    'num_frames': num_frames,
                    'sample_rate': sample_rate
                })
            except:
                print("Error with track", track)

        if not self.track_index:
            raise RuntimeError("No valid tracks found in the given split.")

    def __len__(self) -> int:
        return len(self.track_index)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self._get_item_from_track(self.track_index[idx])