import os
import pickle
from wrappers.phalar_downstream_wrapper import PHALARWrapper
import torch
import torch.nn as nn
from contrastive_model.complex_stuff import CplxLinear
import tqdm
import jams
import numpy as np

RESOLUTION = 0.02

model_path = "/path/to/PHALAR_best.ckpt"
fluidsynth_soundfont = "/path/to/GeneralUser-GS.sf2"
phalar_wrapper = PHALARWrapper(model_path, device='cuda', fluidsynth_soundfont=fluidsynth_soundfont)

class ChordPredictionHead(nn.Module):
    def __init__(self, input_dim=512, output_dim=24):
        super(ChordPredictionHead, self).__init__()
        self.linear = CplxLinear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        # Get magnitude, as chords are phase-invariant
        x = torch.sqrt((x**2).sum(1))
        return x # B x output_dim

# 1. Load Data
AUDIO_DIR = '/path/to/guitarset_audio_mono_mic'
ANNOTATIONS_DIR = '/path/to/guitarset_annotations'

files = list(filter(lambda x: x.endswith('.wav') and '_solo_' not in x, os.listdir(AUDIO_DIR)))

def load_jams_annotations(jams_file):
    """
    Parses JAMS files (standard for GuitarSet) to extract chord annotations.
    """
    jam = jams.load(jams_file)
    
    # Search for chord annotations
    # GuitarSet usually stores them in 'chord' namespace
    anns = jam.search(namespace='chord')
    
    if not anns:
        # Fallback: sometimes namespaces vary, check specifically for any chord-like namespace
        # But usually 'chord' is safe for mir datasets
        print(f"Warning: No chord namespace found in {os.path.basename(jams_file)}")
        return np.array([]), []
        
    # Take the first chord annotation found (usually the ground truth)
    ann = anns[0]
    
    intervals = []
    labels = []
    
    for obs in ann.data:
        # JAMS stores duration, we need end time
        start = obs.time
        end = obs.time + obs.duration
        label = obs.value
        
        intervals.append([start, end])
        labels.append(label)
        
    # convert to segments
    segments = []
    for (start, end), label in zip(intervals, labels):
        segments.append({"start": float(start), "end": float(end), "chord": str(label)})
    return segments

dataset = []
for filename in tqdm.tqdm(files):
    audio_path = os.path.join(AUDIO_DIR, filename)
    with torch.no_grad():
        embeddings = phalar_wrapper._get_audio_embeddings(audio_path, resolution=RESOLUTION)
    embeddings = embeddings.cpu().numpy()
    annotation_name = filename.replace('.wav', '.jams').replace('_mic', '')
    annotation_path = os.path.join(ANNOTATIONS_DIR, annotation_name)

    name = os.path.basename(annotation_path)

    chord_segments = load_jams_annotations(annotation_path)
    for segment in chord_segments:
        start_frame = int(segment['start'] / RESOLUTION)
        end_frame = int(segment['end'] / RESOLUTION)
        chord_label = segment['chord']
        segment_embeddings = embeddings[start_frame:end_frame]
        for emb in segment_embeddings:
            dataset.append((name, emb, chord_label))

# Store as a numpy file for faster loading later
with open('guitarset_chord_embeddings_dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)