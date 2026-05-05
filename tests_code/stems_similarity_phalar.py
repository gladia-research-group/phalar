import os
import sys

# Adds the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torchaudio.functional as F
from pathlib import Path
from contrastive_model.contrastive_model import ContrastiveAudioModelPLWrapper
from contrastive_model.similarity_ops import CplxCosineSimilarity
import librosa
import tqdm
import json

files_dir = '/path/to/stems/folder'
files_dir = Path(files_dir)
model_path = 'PHALAR_best.ckpt'
SAMPLE_RATE = 16000
USE_COS_SIM = False
INSTRUMENTS = ['bass', 'drums']

folders = os.listdir(files_dir)

checkpoint = torch.load(
    model_path,
    map_location="cpu"
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = ContrastiveAudioModelPLWrapper(**checkpoint["hyper_parameters"])
model.load_state_dict(checkpoint["state_dict"], strict=True)
model = model.to(device)
model.eval()

if USE_COS_SIM:
    model.similarity = CplxCosineSimilarity(model.similarity)

results = {"bass": [], "drums": []}

with torch.inference_mode():
    for folder in tqdm.tqdm(folders):
        if folder == 'REMOVED':
            continue
        prefix = files_dir / folder
        for i, instrument in enumerate(INSTRUMENTS):
            submix_without_instrument = prefix / f'{instrument}_input.wav'
            gt = prefix / f'{instrument}_ground_truth.wav'
            sac_result = prefix / f'{instrument}_sac_generated.wav'
            stage_result = prefix / f'{instrument}_stage_generated.wav'
            moises_result = prefix / f'{instrument}_moises_generated.wav'

            paths = [submix_without_instrument, gt, sac_result, stage_result, moises_result]
            waveforms = [torch.from_numpy(librosa.load(path, sr=SAMPLE_RATE)[0]) for path in paths]
            min_len = min([wav.shape[-1] for wav in waveforms])
            waveforms = [wav[..., :min_len] for wav in waveforms]
            waveforms = torch.stack(waveforms)
            
            embeddings = model.encoder(waveforms.to(device))
            simil = model.similarity(embeddings[:1], embeddings[1:])[0].cpu()
            simil += model.similarity(embeddings[1:], embeddings[:1]).T[0].cpu()
            simil /= 2.0
            results[instrument].append(
                {"song": folder, 
                "gt": float(simil[0].item()),
                "sac": float(simil[1].item()),
                "stage": float(simil[2].item()),
                "moises": float(simil[3].item())
                })


# --- Write results to JSON ---
with open("results_phalar.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4)

print(f"Results successfully saved!")