"""CocoChorales Contrastive Torch Dataset (Online Version)."""

from pathlib import Path
import random
import logging

from torchvision.datasets.utils import download_and_extract_archive
import torchaudio
from tqdm import tqdm

from data import OnlineDataset

from contrastive_model import constants

random.seed(14703)


class CocoChoralesContrastivePreprocessed(OnlineDataset):
    """
    CocoChorales Dataset (adapted for online contrastive learning without precomputation).
    Reference: https://magenta.tensorflow.org/datasets/cocochorales
    """

    VERSION = "1.0.0"
    URLS = {
        "train": [f"https://storage.googleapis.com/magentadata/datasets/cocochorales/cocochorales_full_v1_zipped/main_dataset/train/{i}.tar.bz2" for i in [1, 2, 3, 25, 26, 27, 49, 50, 51, 73, 74, 75]],
        "test": [f"https://storage.googleapis.com/magentadata/datasets/cocochorales/cocochorales_full_v1_zipped/main_dataset/test/{i}.tar.bz2" for i in [1, 4, 7, 10]],
        "valid": [f"https://storage.googleapis.com/magentadata/datasets/cocochorales/cocochorales_full_v1_zipped/main_dataset/valid/{i}.tar.bz2" for i in [1, 4, 7, 10]]
    }
    MD5S = {
        "train": [
            "999ba8284b0646a6c7f3ef357e15fd59",
            "f1b6ae484940d66ec006c0599d8b0f48",
            "b2237240c49d3537872d35d98199fdc6",
            "e540dc37fcb47f75995544df3720af3f",
            "7490eb20468f421313bab7882f59c9cf",
            "200eb27e786d27d04347129d10a7731b",
            "358817b12ee126e697f14ef6805cdc48",
            "96e81212eeb8b65619103dd16094a08f",
            "32799d360b9b9764b511d399327509e0",
            "0fa937613c947d0cc18d2d4682504fa0",
            "e5c50a10b0b2af5ee26867c108a94a92",
            "f78dfe2f212e4991a78be7e8e4e98fc5",
        ],
        "test": [
            "2c9e617b9f3ec622e0da35734036af49",
            "461fc00182c5e312ac379d97df4bceb6",
            "f808fc2502059e9a994cea85ccd4d3a0",
            "afebac996cd3d643b7c99d575a3ad048"
        ],
        "valid": [
            "697766f8e53ffc9f64708b8bf4acedb1",
            "4edd6803d082dc090f08823cc003cc94",
            "502128334a38e682ac0a06682207d13b",
            "ded860cabdf005eafde1095ebab7787e",
        ]
    }
    SAMPLE_RATE = 16000
    ORIGINAL_DIR_NAME = "original"

    def __init__(
            self,
            root_dir="~/coco_chorales_contrastive",
            download=True,
            split="train",
            ensemble="random",
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
        self.ensemble = ensemble
        self.samples_per_epoch = samples_per_epoch
        self.device = device

        if self.split not in ["train", "valid", "test"]:
            raise ValueError("`split` must be one of ['train', 'valid', 'test'].")

        if self.ensemble not in ["random", "brass", "string", "woodwind", "*"]:
            raise ValueError("`ensemble` must be one of ['random', 'brass', 'string', 'woodwind', '*'].")

        # Download if not present
        if self.download and not self._is_downloaded_and_extracted():
            self._download_and_extract()

        if not self._is_downloaded_and_extracted():
            raise RuntimeError(
                f"Dataset split {self.split} not found. Please use `download=True` to download it.")
        logging.info(
            f"Found original dataset split {self.split} at {(self.root_dir / self.ORIGINAL_DIR_NAME / self.split)}.")

        # Costruisce l’indice dei brani e degli stems
        self._build_index()

    def _is_downloaded_and_extracted(self) -> bool:
        split_dir = self.root_dir / self.ORIGINAL_DIR_NAME / self.split
        return split_dir.exists() and any(split_dir.iterdir())

    def _download_and_extract(self) -> None:
        for i, url in enumerate(self.URLS[self.split]):
            download_and_extract_archive(
                url, self.root_dir / self.ORIGINAL_DIR_NAME / self.split, md5=self.MD5S[self.split][i], remove_finished=True)

    def _build_index(self):
        original_dir = self.root_dir / self.ORIGINAL_DIR_NAME / self.split
        pattern = f"{self.ensemble}_track*" if self.ensemble != "random" else "*_track*"
        tracks = list(original_dir.glob(pattern))
        if not tracks:
            raise RuntimeError(f"No tracks found for ensemble {self.ensemble} in split {self.split}.")

        self.track_index = []
        for track in tqdm(tracks, desc="Building track index"):
            try:
                stems_paths = list(track.glob("stems_audio/*.wav"))
                if not stems_paths:
                    continue

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
                    'sample_rate': sample_rate
                })
            except:
                print("Error with track", track)

        if not self.track_index:
            raise RuntimeError("No valid tracks found.")

    def __len__(self) -> int:
        return self.samples_per_epoch
