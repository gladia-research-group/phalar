import numpy as np
import torch
import librosa

def mix_down(waveform):
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform

zero_octave_freqs = [
    ('C', 16.35),
    ('C#', 17.32),
    ('D', 18.35),
    ('D#', 19.45),
    ('E', 20.60),
    ('F', 21.83),
    ('F#', 23.12),
    ('G', 24.50),
    ('G#', 25.96),
    ('A', 27.50),
    ('A#', 29.14),
    ('B', 30.87)
]

NOTES, FREQS = zip(*zero_octave_freqs)

def note_name_to_frequency(note_name, octave):
    base_freq = None
    for name, freq in zero_octave_freqs:
        if name == note_name:
            base_freq = freq
            break
    return base_freq * (2 ** octave)

import fluidsynth

def get_chord_notes_with_inversion(root_name: str, base_octave: int, chord_type: str, inversion: int = 0):
    """
    Returns a list of (note_name, is_bass) tuples.
    Example: ('C4', True) means it's the lowest note of that specific voicing.
    """
    CHORD_MAP = {
        "note": [0],
        "maj": [0, 4, 7],
        "min": [0, 3, 7],
        "dim": [0, 3, 6],
        "aug": [0, 4, 8],
        "dom7": [0, 4, 7, 10],
        "maj7": [0, 4, 7, 11],
        "min7": [0, 3, 7, 10],
        "sus2": [0, 2, 7],
        "sus4": [0, 5, 7],
        "maj9": [0, 4, 7, 11, 14],
        "min9": [0, 3, 7, 10, 14],
    }
    
    intervals = CHORD_MAP[chord_type]
    root_idx = NOTES.index(root_name)
    
    # 1. Get raw notes in root position
    notes_in_octaves = []
    for interval in intervals:
        idx = (root_idx + interval) % len(NOTES)
        oct_offset = (root_idx + interval) // len(NOTES)
        notes_in_octaves.append([NOTES[idx], base_octave + oct_offset])

    # 2. Apply Inversion: Shift lower notes up one octave
    # If inversion=1, the first note (Root) moves up.
    # If inversion=2, the first and second notes move up.
    num_notes = len(notes_in_octaves)
    for i in range(inversion % num_notes):
        notes_in_octaves[i][1] += 1
    
    # 3. Sort by pitch to identify the new bass note
    # Sort primarily by octave, then by note index
    notes_in_octaves.sort(key=lambda x: (x[1], NOTES.index(x[0])))
    
    formatted_notes = []
    for i, (name, octv) in enumerate(notes_in_octaves):
        is_bass = (i == 0) # The first note in the sorted list is the bass
        formatted_notes.append((f"{name}{octv}", is_bass))
        
    return formatted_notes

class FluidSynthTemplateGenerator:
    def __init__(self, sf2_path, target_fs=16000):
        self.target_fs = target_fs
        self.internal_fs = 44100  # Standard high-quality internal rate
        
        # 1. Initialize ONE synth correctly
        self.synth = fluidsynth.Synth()
        self.synth.setting('synth.gain', 0.2)
        self.synth.setting('synth.sample-rate', float(self.internal_fs))
        
        # Do NOT use driver="file" when using get_samples; it causes timing conflicts
        self.sfid = self.synth.sfload(sf2_path)
        
        self.instruments = {
            "piano": 0,
            "guitar": 24,
            "electric_guitar": 26,
            "bass": 32,
            "electric_bass": 34,
        }

    def _note_to_midi(self, note_name_with_octave):
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        name = note_name_with_octave[:-1]
        octave = int(note_name_with_octave[-1])
        return notes.index(name) + (octave + 1) * 12

    def generate_master_template(
        self, root, chord_type, inversion, 
        duration_seconds=2.0, 
        instrument="piano", octaves=[3, 4, 5], 
        add_cents=0
    ):
        # 2. Correct Sample Calculation (Stereo = Frames * 2)
        total_frames = int(self.internal_fs * duration_seconds)
        total_samples_to_get = total_frames * 2 
        
        self.synth.program_select(0, self.sfid, 0, self.instruments.get(instrument, 0))
        
        # # 3. Apply Pitch Bend (Range 0-16383, 8192 is Center)
        # # We divide by 200 because default MIDI bend range is +/- 2 semitones (200 cents)
        self.synth.pitch_bend(0, 0) # 0 is Center in pyfluidsynth
        
        # 2. Updated Math: Center is 0
        # Range is -8192 to 8191. We'll use 8191 as the multiplier for the up-swing.
        shift_factor = 8191.0 if add_cents >= 0 else 8192.0
        bend_value = int((add_cents / 200.0) * shift_factor)
        
        # Clamp to the wrapper's limits
        bend_value = max(-8192, min(8191, bend_value))
        
        self.synth.pitch_bend(0, bend_value)

        # 4. Trigger Notes
        for octv in octaves:
            # Note: Ensure your get_chord_notes_with_inversion logic is accessible here
            notes_data = get_chord_notes_with_inversion(root, octv, chord_type, inversion)
            for note_name, is_bass in notes_data:
                midi_note = self._note_to_midi(note_name)
                velocity = 110 if is_bass else 90
                self.synth.noteon(0, midi_note, velocity)
        
        # 5. Render
        raw_samples = self.synth.get_samples(total_samples_to_get)
        
        # 6. CRITICAL: Cleanup and RESET for the next call
        self.synth.all_notes_off(0)
        self.synth.pitch_bend(0, 0) # Reset bend to neutral center

        # 7. Convert and Downmix
        audio = np.array(raw_samples, dtype=np.float32) / 32768.0
        # Ensure we reshape correctly: (frames, channels)
        mono_waveform_44100 = audio.reshape(-1, 2).mean(axis=1)

        # 8. High-Quality Resample
        mono_waveform_16000 = librosa.resample(
            mono_waveform_44100, 
            orig_sr=self.internal_fs, 
            target_sr=self.target_fs
        )

        # 9. Normalize
        max_val = np.max(np.abs(mono_waveform_16000))
        if max_val > 0:
            mono_waveform_16000 = (mono_waveform_16000 / max_val) * 0.8
            
        return mono_waveform_16000

    def generate_metronome(self, bpm, duration_seconds, offset_s=0.02, midi_note=115):
        """
        Generates a metronome track.
        :param bpm: Beats per minute
        :param duration_seconds: Total length of the audio
        :param offset_s: Delay before the first beat in seconds
        :param midi_note: MIDI note to use (115 is Woodblock in GM)
        """
        beat_interval = 60.0 / bpm
        samples_per_beat = int(self.internal_fs * beat_interval)
        offset_samples = int(self.internal_fs * offset_s)
        total_frames = int(self.internal_fs * duration_seconds)
        
        # Ensure we use a percussive-friendly instrument
        self.synth.program_select(0, self.sfid, 0, midi_note)
        
        full_audio_stereo = []

        # 1. Handle Offset (Silence before first beat)
        if offset_samples > 0:
            silence = self.synth.get_samples(offset_samples)
            full_audio_stereo.extend(silence)

        # 2. Generate Beats
        current_frame = offset_samples
        while current_frame < total_frames:
            # Trigger the click
            self.synth.noteon(0, self._note_to_midi("A4"), 120)
            
            # Determine how many samples to render for this beat
            # We don't want to exceed the total duration
            remaining_frames = total_frames - current_frame
            frames_to_render = min(samples_per_beat, remaining_frames)
            
            beat_samples = self.synth.get_samples(frames_to_render)
            full_audio_stereo.extend(beat_samples)
            
            # Clean up note to prevent bleeding if the interval is short
            self.synth.noteoff(0, midi_note)
            
            current_frame += frames_to_render
            if frames_to_render < samples_per_beat:
                break

        # 3. Post-processing (Same logic as your chord generator)
        audio = np.array(full_audio_stereo, dtype=np.float32) / 32768.0
        mono_waveform_44100 = audio.reshape(-1, 2).mean(axis=1)

        # Resample to target FS
        mono_waveform_16000 = librosa.resample(
            mono_waveform_44100, 
            orig_sr=self.internal_fs, 
            target_sr=self.target_fs
        )

        # Normalize
        max_val = np.max(np.abs(mono_waveform_16000))
        if max_val > 0:
            mono_waveform_16000 = (mono_waveform_16000 / max_val) * 0.8
            
        return mono_waveform_16000

# Example usage
if __name__ == "__main__":
    import soundfile as sf

    fs_gen = FluidSynthTemplateGenerator("/path/to/GeneralUser-GS.sf2", target_fs=16000)

    bpm = 240
    metronome_waveform = fs_gen.generate_metronome(bpm, duration_seconds=5.0)
    sf.write('metronome.wav', metronome_waveform, 16000)

    chord_waveform = fs_gen.generate_master_template(
        root="E", chord_type="note", inversion=0, 
        duration_seconds=5.0, instrument="electric_bass", octaves=[3]
    )
    sf.write('chord_template.wav', chord_waveform, 16000)

    # 5. Detect Pitch on the 16k audio
    freqs = librosa.yin(y=chord_waveform, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'), sr=16000)
    avg_freq = np.mean(freqs)

    print(f"Detected: {librosa.hz_to_note(avg_freq)} ({avg_freq:.2f} Hz)")