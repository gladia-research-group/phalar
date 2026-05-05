import os
import sys

# Adds the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from lightning.pytorch.cli import LightningCLI
from contrastive_model.contrastive_model_data import DataModule
from wrappers.cdpam_wrapper import CDPAMComparisonPLWrapper

import logging

torch.set_float32_matmul_precision('medium')
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

import multiprocessing
# Set thread creation to spawn to avoid issues with dataloaders and CUDA
if multiprocessing.get_start_method(allow_none=True) != 'forkserver':
    multiprocessing.set_start_method('forkserver', force=True)


def cli_main():
    cli = LightningCLI(CDPAMComparisonPLWrapper, DataModule, save_config_callback=None)


if __name__ == "__main__":
    cli_main()
