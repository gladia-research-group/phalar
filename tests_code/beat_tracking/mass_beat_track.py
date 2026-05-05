import os
import sys

# Adds the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tqdm
import json
import torch
import random
from pathlib import Path
from beat_this.inference import File2Beats
from wrappers.phalar_downstream_wrapper import PHALARWrapper

# Prepare
random.seed(42)
torch.manual_seed(42)
output_folder = Path(".hidden/beats")
output_folder.mkdir(exist_ok=True) # Create folder if it doesn't exist

file2beats = File2Beats(checkpoint_path="final0", device="cuda", dbn=False)

model_path = "PHALAR_best.ckpt"
fluidsynth_soundfont = None
device = "cuda:0"

whatever = PHALARWrapper(
    model_path=model_path,
    device=device,
    fluidsynth_soundfont=fluidsynth_soundfont
)

mp3_files = list(Path(".hidden/gtzan/genres_original").glob("*/*.wav"))


# Process files
for audio_path in tqdm.tqdm(mp3_files, desc="Processing audio files"):
    # Check if output already exists
    output_path = output_folder / f"{audio_path.stem}.json"
    if output_path.exists():
        print(f"\nSkipping {audio_path.name}, output already exists.")
        continue
    try:
        beats, downbeats = file2beats(str(audio_path))

        our_beats, bpm_estimate = whatever.track_beats(str(audio_path))

        # Save results
        with open(output_path, "w") as f:
            json.dump({
                "file2beats_beats": beats.tolist(),
                "file2beats_downbeats": downbeats.tolist(),
                "our_beats": our_beats.tolist(),
                "bpm_estimate": bpm_estimate
            }, f, indent=4)
    except Exception as e:
        print(f"\nError processing {audio_path.name}: {e}")

print("Done!")