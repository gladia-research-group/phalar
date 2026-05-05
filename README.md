# PHALAR: Phasors for Learned Musical Audio Representations

**PHALAR** is a self-supervised contrastive learning framework for learning rich embeddings of musical audio. Our key innovation is a **learned spectral pooling technique** that feeds into a **phase-equivariant complex-valued neural network (CVNN)**, enabling the model to learn phase-aware representations of music.

This repository contains the official implementation of the PHALAR model and related baselines (COCOLA, MERT-based variants) with support for multiple audio datasets and downstream task evaluation.
## Installation

### Requirements

- Python 3.12
- PyTorch 2.7 with CUDA support (or CPU)
- FluidSynth with GeneralUser-GS soundfont (for template synthesis)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/gladia-research-group/phalar.git
   cd phalar
   ```

2. **Install dependencies**:
   Using `uv` (recommended for speed):
   ```bash
   uv sync
   ```
   
   Or with `pip`:
   ```bash
   pip install -e .
   ```

3. **Install FluidSynth** (for synthesized template generation):
   ```bash
   # macOS
   brew install fluidsynth
   
   # Ubuntu/Debian
   sudo apt-get install fluidsynth
   
   # Fedora
   sudo dnf install fluidsynth
   ```

4. **Download soundfont**:
   The repository includes `GeneralUser-GS.sf2` for MIDI synthesis. Ensure this file is accessible for template generation.

## Quick Start

### Load Pre-trained Models

```python
import torch
from contrastive_model.contrastive_model import ContrastiveAudioModelPLWrapper

# Load PHALAR model
model_path = 'ckpts/PHALAR_best.ckpt'
checkpoint = torch.load(model_path, map_location="cpu")

model = ContrastiveAudioModelPLWrapper(**checkpoint["hyper_parameters"])
model.load_state_dict(checkpoint["state_dict"], strict=True)
model.eval()

# Make similarity commutative (for phase-equivariant models)
model.similarity.hermitian = True
```

### Generate Embeddings

```python
# Create audio tensor (16 kHz mono audio)
audio = torch.randn(1, 1, 5 * 16000)  # 5 seconds of audio

with torch.inference_mode():
    embedding = model.encoder(audio)
    print(f"Embedding shape: {embedding.shape}")
```

### Compute Similarity

```python
audio1 = torch.randn(1, 1, 5 * 16000)
audio2 = torch.randn(1, 1, 7 * 16000)

with torch.inference_mode():
    emb1 = model.encoder(audio1)
    emb2 = model.encoder(audio2)
    
    # Compute symmetric similarity
    score = model.similarity(emb1, emb2)
    print(f"Similarity: {score.item():.4f}")
```

For full inference examples (including COCOLA), see [model_usage_example.py](model_usage_example.py).

## Training

Training uses PyTorch Lightning CLI with YAML configuration files (be sure to download and prepare your data! To do so see [data/])

```bash
uv run python train.py fit \
    --config configs/train_phalar.yaml \
    --trainer.max_epochs 100 \
    --trainer.devices 1
```

Different training configurations are in `configs/`.

## Inference & Downstream Tasks

### Using Pre-trained Embeddings

The `wrappers/` folder contains utilities for downstream task evaluation:

```python
from wrappers.phalar_downstream_wrapper import PHALARWrapper

wrapper = PHALARWrapper(
    model_path='ckpts/PHALAR_best.ckpt',
    device='cuda',
    fluidsynth_soundfont='GeneralUser-GS.sf2'
)

# Get embedding for audio file
embedding = wrapper.get_embedding('audio.wav')

# Use for downstream tasks
# Examples in tests_code/: chord detection, beat tracking
```

See `wrappers/` for:
- [phalar_downstream_wrapper.py](wrappers/phalar_downstream_wrapper.py) – PHALAR wrapper
- [clap_wrapper.py](wrappers/clap_wrapper.py) – CLAP baseline
- [cdpam_wrapper.py](wrappers/cdpam_wrapper.py) – CDPAM baseline

### Downstream Evaluation Scripts

- **Chord Detection**: [tests_code/chord_detection/](tests_code/chord_detection/)
- **Beat Tracking**: [tests_code/beat_tracking/](tests_code/beat_tracking/)
- **Stem Similarity**: [tests_code/stems_similarity_*.py](tests_code/)

## Citation
If you use PHALAR in your research, please cite:

```bibtex
@inproceedings{marincione2026phalar,
    title={PHALAR: Phasors for Learned Musical Audio Representations}, 
    author={Davide Marincione and Michele Mancusi and Giorgio Strano and Luca Cerovaz and Roberto Ribuoli and Emanuele Rodol{\`a}},
    year={2026},
    booktitle={Proceedings of the Forty-Third International Conference on Machine Learning}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments
- **European Research Council**, for the funds that enabled this work.
- **ISCRA/CINECA**, for the access to the Leonardo cluster, that powered the training.
- **User study participants**, for the invaluable listening tests.
