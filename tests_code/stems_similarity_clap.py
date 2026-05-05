import os
import sys

# Adds the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torchaudio.functional as F
from pathlib import Path
from wrappers.clap_wrapper import CLAPComparisonPLWrapper
import librosa
import tqdm
import json

files_dir = '/path/to/stems/folder'
files_dir = Path(files_dir)
SAMPLE_RATE = 16000
INSTRUMENTS = ['bass', 'drums']

folders = os.listdir(files_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CLAPComparisonPLWrapper()
model = model.to(device)
model.eval()

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
            
            embeddings = model._get_clap_embeddings(waveforms.to(device))
            simil = model.similarity(embeddings[:1], embeddings[1:])[0].cpu()
            results[instrument].append(
                {"song": folder, 
                "gt": float(simil[0].item()),
                "sac": float(simil[1].item()),
                "stage": float(simil[2].item()),
                "moises": float(simil[3].item())
                })


# --- Write results to JSON ---
with open("results_clap.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4)

print(f"Results successfully saved!")