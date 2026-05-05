import os
import glob
import librosa
import soundfile as sf
from tqdm import tqdm

def sample_musdb_mixtures(musdb_root, output_dir, duration=10.0, offset=10.0):
    """
    Finds all mixture.wav files in MUSDB and extracts a 10s snippet.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    search_path = os.path.join(musdb_root, "**", "mixture.wav")
    mixtures = glob.glob(search_path, recursive=True)
    
    if not mixtures:
        print(f"No mixture.wav files found in {musdb_root}. Check your path!")
        return

    print(f"Found {len(mixtures)} tracks. Extracting {duration}s snippets...")

    for path in tqdm(mixtures):
        # Create a clean filename: "Artist - Title.wav"
        # In MUSDB, the parent folder name is usually the song title
        folder_name = os.path.basename(os.path.dirname(path))
        output_path = os.path.join(output_dir, f"{folder_name}.wav")

        try:
            # Load only the specified snippet
            # MUSDB is 44100Hz by default
            y, sr = librosa.load(path, sr=44100, offset=offset, duration=duration)
            
            # Save as high-quality WAV
            sf.write(output_path, y, sr)
        except Exception as e:
            print(f"Skipping {folder_name} due to error: {e}")

# --- CONFIG ---
MUSDB_PATH = '/input/musdb/path' 
OUTPUT_PATH = '/output/musdb/samples/path'

sample_musdb_mixtures(MUSDB_PATH, OUTPUT_PATH)