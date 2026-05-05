"""
Lightning DataModules.
"""

import os
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader, ConcatDataset
import torch
from typing import List

from contrastive_model import constants
from feature_extraction.feature_extraction import FeatureExtractor
#from data.coco_chorales_contrastive_preprocessed import CocoChoralesContrastivePreprocessed
from data.coco_chorales_contrastive_online import CocoChoralesContrastivePreprocessed
from data.moisesdb_contrastive_online import MoisesdbContrastivePreprocessed
from data.slakh2100_contrastive_online import Slakh2100ContrastivePreprocessed
#from data.moisesdb_contrastive_preprocessed import MoisesdbContrastivePreprocessed
#from data.slakh2100_contrastive_preprocessed import Slakh2100ContrastivePreprocessed
from data.coco_chorales_contrastive_exclusive import CocoChoralesContrastiveExclusive
from data.moisesdb_contrastive_exclusive import MoisesdbContrastiveExclusive
from data.slakh2100_contrastive_exclusive import Slakh2100ContrastiveExclusive

class RandomCropCollator:
    def __init__(self, min_ratio, range_size):
        self.min_ratio = min_ratio
        self.range_size = range_size

    def __call__(self, batch):
        r = torch.rand(1).item() * self.range_size + self.min_ratio
        tot_L = batch[0]["positive"].shape[-1]
        L = int(tot_L * r)

        positives = []
        anchors = []
        for item in batch:
            if L < tot_L:
                offset = torch.randint(0, tot_L - L, ()).item()
            else:
                offset = 0
            positives.append(item["positive"][..., offset:offset + L])
            anchors.append(item["anchor"][..., offset:offset + L])

        return {
            "positive": torch.stack(positives),
            "anchor": torch.stack(anchors),
        }

class DataModule(L.LightningDataModule):
    def __init__(self,
                 root_dir: str = "~",
                 dataset: constants.Dataset = constants.Dataset.CCS,
                 batch_size: int = 32,
                 chunk_duration_range: List[int] = None,
                 chunk_duration_test: int = 5,
                 target_sample_rate: int = 16000,
                 generate_submixtures: bool = True,
                 feature_extractor_type: constants.ModelFeatureExtractorType = constants.ModelFeatureExtractorType.HPSS,
                 feature_extraction_time: constants.FeatureExtractionTime = constants.FeatureExtractionTime.OFFLINE,
                 augmentations: dict = dict(),
                 downsample_mono: bool = True,
                 n_mels: int = 128,
                 ):
        super().__init__()
        self.save_hyperparameters()
        
        self.root_dir = Path(root_dir)
        self.dataset = dataset
        self.batch_size = batch_size
        self.chunk_duration_range = chunk_duration_range
        self.chunk_duration_test = chunk_duration_test
        self.target_sample_rate = target_sample_rate
        self.generate_submixtures = generate_submixtures
        self.feature_extractor_type = feature_extractor_type
        self.feature_extraction_time = feature_extraction_time
        self.augmentations = augmentations
        self.downsample_mono = downsample_mono
        self.chunk_duration_train_dataset = chunk_duration_range[-1] if chunk_duration_range is not None else chunk_duration_test
        self.n_mels = n_mels

        feature_extractor = FeatureExtractor(
            feature_extractor_type=self.feature_extractor_type,
            n_mels=self.n_mels)
        self.preprocess_transform = None
        self.runtime_transform = None
        if self.feature_extraction_time == constants.FeatureExtractionTime.OFFLINE:
            self.preprocess_transform = feature_extractor
        elif self.feature_extraction_time == constants.FeatureExtractionTime.ONLINE:
            self.runtime_transform = feature_extractor

    def setup(self, stage: str):
        if self.dataset in {constants.Dataset.CCS,
                            constants.Dataset.CCS_RANDOM,
                            constants.Dataset.CCS_STRING,
                            constants.Dataset.CCS_BRASS,
                            constants.Dataset.CCS_WOODWIND}:
            ensemble = self.dataset.value.split("/")[1]
            self.train_dataset, self.val_dataset, self.test_dataset = self._get_cocochorales_splits(
                ensemble=ensemble, stage=stage)

        elif self.dataset == constants.Dataset.SLAKH2100:
            self.train_dataset, self.val_dataset, self.test_dataset = self._get_slakh2100_splits(
                stage)

        elif self.dataset == constants.Dataset.MOISESDB:
            self.train_dataset, self.val_dataset, self.test_dataset = self._get_moisesdb_splits(
                stage)
        elif self.dataset == constants.Dataset.MOISESDB_EXCLUSIVE:
            self.train_dataset, self.val_dataset, self.test_dataset = self._get_moisesdb_exclusive_splits(
                stage)

        elif self.dataset == constants.Dataset.MIXED:
            self.train_dataset, self.val_dataset, self.test_dataset = self._get_mixed_splits(
                stage)

    def _get_mixed_splits(self, stage: str):
        slakh_train_dataset, slakh_val_dataset, slakh_test_dataset = self._get_slakh2100_mixed_splits(
            stage)
        coco_train_dataset, coco_val_dataset, coco_test_dataset = self._get_cocochorales_mixed_splits(
            "random", stage=stage)
        moisesdb_train_dataset, moisesdb_val_dataset, moisesdb_test_dataset = self._get_moisesdb_mixed_splits(
            stage)

        train_dataset, val_dataset, test_dataset = None, None, None
        if stage == 'fit':
            train_dataset = ConcatDataset(
                [coco_train_dataset, moisesdb_train_dataset, slakh_train_dataset])
            val_dataset = ConcatDataset(
                [coco_val_dataset, moisesdb_val_dataset, slakh_val_dataset])
        elif stage == 'test':
            test_dataset = ConcatDataset(
                [coco_test_dataset, moisesdb_test_dataset, slakh_test_dataset])

        return train_dataset, val_dataset, test_dataset

    def _get_cocochorales_splits(self, ensemble: str, stage: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        root_dir = self.root_dir / "coco_chorales_contrastive"
        train_dataset, val_dataset, test_dataset = None, None, None

        if stage == "fit":
            train_dataset = CocoChoralesContrastivePreprocessed(
                root_dir=root_dir,
                download=True,
                #preprocess=True,
                split="train",
                ensemble=ensemble,
                chunk_duration=self.chunk_duration_train_dataset,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                augmentations=self.augmentations,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )

            val_dataset = CocoChoralesContrastivePreprocessed(
                root_dir=root_dir,
                download=True,
                #preprocess=True,
                split="valid",
                ensemble=ensemble,
                chunk_duration=self.chunk_duration_test,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )
        elif stage == "test":
            test_dataset = CocoChoralesContrastivePreprocessed(
                root_dir=root_dir,
                download=True,
                #preprocess=True,
                split="test",
                ensemble=ensemble,
                chunk_duration=self.chunk_duration_test,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )

        return train_dataset, val_dataset, test_dataset

    def _get_cocochorales_exclusive_splits(self, ensemble: str, stage: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        root_dir = self.root_dir / "coco_chorales_contrastive"
        train_dataset, val_dataset, test_dataset = None, None, None

        if stage == "fit":
            train_dataset = CocoChoralesContrastiveExclusive(
                root_dir=root_dir,
                download=True,
                #preprocess=True,
                split="train",
                ensemble=ensemble,
                chunk_duration=self.chunk_duration_train_dataset,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                augmentations=self.augmentations,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )

            val_dataset = CocoChoralesContrastivePreprocessed(
                root_dir=root_dir,
                download=True,
                #preprocess=True,
                split="valid",
                ensemble=ensemble,
                chunk_duration=self.chunk_duration_test,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )
        elif stage == "test":
            test_dataset = CocoChoralesContrastivePreprocessed(
                root_dir=root_dir,
                download=True,
                #preprocess=True,
                split="test",
                ensemble=ensemble,
                chunk_duration=self.chunk_duration_test,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )

        return train_dataset, val_dataset, test_dataset

    def _get_cocochorales_mixed_splits(self, ensemble: str, stage: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        root_dir = self.root_dir / "coco_chorales_contrastive"
        train_dataset, val_dataset, test_dataset = None, None, None

        if stage == "fit":
            train_dataset = CocoChoralesContrastivePreprocessed(
                root_dir=root_dir,
                download=True,
                #preprocess=True,
                split="train",
                ensemble=ensemble,
                chunk_duration=self.chunk_duration_train_dataset,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                samples_per_epoch=3333,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                augmentations=self.augmentations,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )

            val_dataset = CocoChoralesContrastivePreprocessed(
                root_dir=root_dir,
                download=True,
                #preprocess=True,
                split="valid",
                ensemble=ensemble,
                chunk_duration=self.chunk_duration_test,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                samples_per_epoch=3333,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )
        elif stage == "test":
            test_dataset = CocoChoralesContrastivePreprocessed(
                root_dir=root_dir,
                download=True,
                #preprocess=True,
                split="test",
                ensemble=ensemble,
                chunk_duration=self.chunk_duration_test,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                samples_per_epoch=3333,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )

        return train_dataset, val_dataset, test_dataset

    def _get_slakh2100_splits(self, stage: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        root_dir = self.root_dir / "slakh2100_contrastive"
        train_dataset, val_dataset, test_dataset = None, None, None

        if stage == "fit":
            train_dataset = Slakh2100ContrastivePreprocessed(
                root_dir=root_dir,
                download=True,
                #preprocess=True,
                split="train",
                chunk_duration=self.chunk_duration_train_dataset,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                augmentations=self.augmentations,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )
            val_dataset = Slakh2100ContrastivePreprocessed(
                root_dir=root_dir,
                download=True,
                #preprocess=True,
                split="validation",
                chunk_duration=self.chunk_duration_test,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )
        elif stage == "test":
            test_dataset = Slakh2100ContrastivePreprocessed(
                root_dir=root_dir,
                download=True,
                #preprocess=True,
                split="test",
                chunk_duration=self.chunk_duration_test,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )

        return train_dataset, val_dataset, test_dataset

    def _get_slakh2100_exclusive_splits(self, stage: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        root_dir = self.root_dir / "slakh2100_contrastive"
        train_dataset, val_dataset, test_dataset = None, None, None

        if stage == "fit":
            train_dataset = Slakh2100ContrastiveExclusive(
                root_dir=root_dir,
                download=True,
                #preprocess=True,
                split="train",
                chunk_duration=self.chunk_duration_train_dataset,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                augmentations=self.augmentations,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )
            val_dataset = Slakh2100ContrastivePreprocessed(
                root_dir=root_dir,
                download=True,
                #preprocess=True,
                split="validation",
                chunk_duration=self.chunk_duration_test,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )
        elif stage == "test":
            test_dataset = Slakh2100ContrastivePreprocessed(
                root_dir=root_dir,
                download=True,
                #preprocess=True,
                split="test",
                chunk_duration=self.chunk_duration_test,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )

        return train_dataset, val_dataset, test_dataset

    def _get_slakh2100_mixed_splits(self, stage: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        root_dir = self.root_dir / "slakh2100_contrastive"
        train_dataset, val_dataset, test_dataset = None, None, None

        if stage == "fit":
            train_dataset = Slakh2100ContrastivePreprocessed(
                root_dir=root_dir,
                download=True,
                #preprocess=True,
                split="train",
                chunk_duration=self.chunk_duration_train_dataset,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                samples_per_epoch=3333,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                augmentations=self.augmentations,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )
            val_dataset = Slakh2100ContrastivePreprocessed(
                root_dir=root_dir,
                download=True,
                #preprocess=True,
                split="validation",
                chunk_duration=self.chunk_duration_test,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                samples_per_epoch=3333,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )
        elif stage == "test":
            test_dataset = Slakh2100ContrastivePreprocessed(
                root_dir=root_dir,
                download=True,
                #preprocess=True,
                split="test",
                chunk_duration=self.chunk_duration_test,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                samples_per_epoch=3333,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )

        return train_dataset, val_dataset, test_dataset

    def _get_moisesdb_splits(self, stage: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        root_dir = self.root_dir / "moisesdb_contrastive"
        train_dataset, val_dataset, test_dataset = None, None, None

        if stage == "fit":
            train_dataset = MoisesdbContrastivePreprocessed(
                root_dir=root_dir,
                split="train",
                #preprocess=True,
                chunk_duration=self.chunk_duration_train_dataset,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                augmentations=self.augmentations,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )

            val_dataset = MoisesdbContrastivePreprocessed(
                root_dir=root_dir,
                split="valid",
                #preprocess=True,
                chunk_duration=self.chunk_duration_test,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )
        elif stage == "test":
            test_dataset = MoisesdbContrastivePreprocessed(
                root_dir=root_dir,
                split="test",
                #preprocess=True,
                chunk_duration=self.chunk_duration_test,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )

        return train_dataset, val_dataset, test_dataset
    
    def _get_moisesdb_exclusive_splits(self, stage: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        root_dir = self.root_dir / "moisesdb_contrastive"
        train_dataset, val_dataset, test_dataset = None, None, None

        if stage == "fit":
            train_dataset = MoisesdbContrastiveExclusive(
                root_dir=root_dir,
                split="train",
                #preprocess=True,
                chunk_duration=self.chunk_duration_train_dataset,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                augmentations=self.augmentations,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )

            val_dataset = MoisesdbContrastivePreprocessed(
                root_dir=root_dir,
                split="valid",
                #preprocess=True,
                chunk_duration=self.chunk_duration_test,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )
        elif stage == "test":
            test_dataset = MoisesdbContrastivePreprocessed(
                root_dir=root_dir,
                split="test",
                #preprocess=True,
                chunk_duration=self.chunk_duration_test,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )

        return train_dataset, val_dataset, test_dataset

    def _get_moisesdb_mixed_splits(self, stage: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        root_dir = self.root_dir / "moisesdb_contrastive"
        train_dataset, val_dataset, test_dataset = None, None, None

        if stage == "fit":
            train_dataset = MoisesdbContrastivePreprocessed(
                root_dir=root_dir,
                split="train",
                #preprocess=True,
                chunk_duration=self.chunk_duration_train_dataset,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                samples_per_epoch=3334,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                augmentations=self.augmentations,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )

            val_dataset = MoisesdbContrastivePreprocessed(
                root_dir=root_dir,
                split="valid",
                #preprocess=True,
                chunk_duration=self.chunk_duration_test,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                samples_per_epoch=3334,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )
        elif stage == "test":
            test_dataset = MoisesdbContrastivePreprocessed(
                root_dir=root_dir,
                split="test",
                #preprocess=True,
                chunk_duration=self.chunk_duration_test,
                target_sample_rate=self.target_sample_rate,
                generate_submixtures=self.generate_submixtures,
                device=device,
                samples_per_epoch=3334,
                preprocess_transform=self.preprocess_transform,
                runtime_transform=self.runtime_transform,
                feature_extractor_type=self.feature_extractor_type,
                mono=self.downsample_mono
                )

        return train_dataset, val_dataset, test_dataset

    def train_dataloader(self):
        if self.chunk_duration_range is None or self.chunk_duration_range[-1] is None:
            collate_fn = None
        else:
            min_len, max_len = self.chunk_duration_range[0], self.chunk_duration_range[1]
            assert min_len <= max_len, "min_len can't be higher than max_len"
            min_ratio = min_len / max_len
            range_size = 1.0 - min_ratio
            collate_fn = RandomCropCollator(min_ratio, range_size)
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=os.cpu_count() - 1,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=os.cpu_count() - 1,
            pin_memory=True,
            persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=os.cpu_count() - 1,
            pin_memory=True,
            persistent_workers=True)
