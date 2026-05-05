import torchaudio
from tqdm import tqdm
from pathlib import Path
import logging
from data import OnlineDataset

from contrastive_model import constants

class MoisesdbContrastivePreprocessed(OnlineDataset):
    """
    Moisesdb Dataset with random overlap between chunks for contrastive learning.
    """

    VERSION = "1.0.0"
    SAMPLE_RATE = 44100
    ORIGINAL_DIR_NAME = "moisesdb_v0.1"

    def __init__(
            self,
            root_dir="~/moisesdb_contrastive",
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
        self.split = split
        self.device = device
        self.samples_per_epoch = samples_per_epoch  # Total number of samples per epoch

        if self.split not in ["train", "valid", "test"]:
            raise ValueError(
                "`split` must be one of ['train', 'valid', 'test'].")

        if not self._is_downloaded_and_extracted():
            raise RuntimeError(
                f"Dataset split {self.split} not found.")
        logging.info(
            f"Found original dataset split {self.split} at {(self.root_dir / self.ORIGINAL_DIR_NAME / self.split)}.")

        self._build_index()

    def _is_downloaded_and_extracted(self) -> bool:
        split_dir = self.root_dir / self.ORIGINAL_DIR_NAME / self.split
        return split_dir.exists() and any(split_dir.iterdir())

    def _build_index(self):
        original_dir = self.root_dir / self.ORIGINAL_DIR_NAME / self.split
        tracks = list(original_dir.glob("*"))
        if not tracks:
            raise RuntimeError(f"No tracks found in split {self.split}.")

        self.track_index = []
        for track in tqdm(tracks, desc="Building track index"):
            stems_paths = list(track.glob("*/*.wav"))
            if not stems_paths:
                continue

            # Get total number of frames (assuming all stems have the same duration)
            # THIS ASSUMPTION IS WRONG!!!! At least the sr's are equal for all T.T
            lengths = []
            for stem in stems_paths:
                info = torchaudio.info(stem, backend="soundfile")
                lengths.append(info.num_frames)

            info = torchaudio.info(str(stems_paths[0]))
            num_frames = min(lengths)
            sample_rate = info.sample_rate

            self.track_index.append({
                'track_name': track.name,
                'stems_paths': stems_paths,
                'num_frames': num_frames,
                'sample_rate': sample_rate,
            })

        if not self.track_index:
            raise RuntimeError(f"No valid tracks found in split {self.split}.")

    def __len__(self) -> int:
        return self.samples_per_epoch  # Define the number of samples per epoch
