from lightning.pytorch.cli import LightningCLI
import torch
from contrastive_model.contrastive_model_data import DataModule
from contrastive_model.contrastive_model import ContrastiveAudioModelPLWrapper

import logging

torch.set_float32_matmul_precision('medium')
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

import multiprocessing
# Set thread creation to spawn to avoid issues with dataloaders and CUDA
if multiprocessing.get_start_method(allow_none=True) != 'forkserver':
    multiprocessing.set_start_method('forkserver', force=True)


def cli_main():
    cli = LightningCLI(ContrastiveAudioModelPLWrapper, DataModule, save_config_callback=None)


if __name__ == "__main__":
    cli_main()
