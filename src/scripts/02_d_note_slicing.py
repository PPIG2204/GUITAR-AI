import os
import glob
import jams
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import sys

# Centralized Path Management
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from paths import HEX_AUDIO, ANNOTATION, ATOM_DIR
except ImportError:
    print("❌ paths.py not found. Ensure it is in the parent or current directory.")
    exit()

# =========================
# CONFIG
# =========================
AUDIO_DIR = str(HEX_AUDIO)      # Using hex-pickup for signal purity
JAMS_DIR  = str(ANNOTATION)
OUTPUT_DIR = str(ATOM_DIR)
SAMPLE_RATE = 22050
AUGMENT_HIGH_FRETS = True 
MAX_ATOM_DURATION = 1.2        # Truncate overly long ring-outs
ENERGY_THRESHOLD = 0.01        # Minimum RMS energy to be considered a valid note

# MIDI Base for Open Strings (Standard Tuning)
OPEN_STRING_MIDI = [40, 45, 50, 55, 59, 64]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_zero_crossing(audio, idx, search_window=100):
    if idx < 0: return 0
    if idx >= len(audio): return len(audio) - 1
    start = max(0, idx - search_window)
    end = min(len(audio), idx + search_window)
    window = audio[start:end]
    zero_crossings = np.where(np.diff(np.sign(window)))[0]
    if len(zero_crossings) == 0: return idx
    closest_local = zero_crossings[np.abs(zero_crossings - (idx - start)).argmin()]
    return start + closest_local

def process_file(jam_path, audio_path):
    try:
        jam = jams.load(jam_path)
        # Load as 6-channel multi-track
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=False)
    except Exception as e:
        print(f"⚠️ Error loading {os.path.basename(audio_path)}: {e}")
        return

    for ann in jam.annotations:
        if ann.namespace != 'note_midi':
            continue
            
        try:
            string_num = int(ann.annotation_metadata.data_source)
            # Extract pure signal from the specific hex-pickup channel
            string_audio = y[string_num]
        except:
            continue

        for obs in ann.data:
            midi_pitch = obs.value
            start_time = obs.time
            # Limit duration to prevent "two notes" in one slice
            duration = min(obs.duration, MAX_ATOM_DURATION)
            
            if duration < 0.1: continue
            
            ref_midi = OPEN_STRING_MIDI[string_num]
            fret = int(round(midi_pitch - ref_midi))
            
            if fret < 0 or fret > 24: continue 

            # Calculate indices
            start_idx = int(start_time * sr)
            end_idx = int((start_time + duration) * sr)
            
            # Anti-Click
            start_idx = find_zero_crossing(string_audio, start_idx)
            end_idx = find_zero_crossing(string_audio, end_idx)
            
            atom = string_audio[start_idx : end_idx]
            
            # VALIDATION: Energy Check (Garbage Out prevention)
            rms = np.sqrt(np.mean(atom**2))
            if rms < ENERGY_THRESHOLD or len(atom) < 500:
                continue

            # Apply Fade for smooth transitions
            fade_len = int(0.01 * sr)
            if len(atom) > 2 * fade_len:
                atom[:fade_len] *= np.linspace(0, 1, fade_len)
                atom[-fade_len:] *= np.linspace(1, 0, fade_len)

            # Save Base Atom
            base_name = os.path.basename(audio_path).replace(".wav", "")
            save_name = f"str{string_num}_fr{fret}_{base_name}_{start_idx}.wav"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            sf.write(save_path, atom, sr)
            
            # Synthetic Generation for High Fret Balancing
            if AUGMENT_HIGH_FRETS and fret < 8:
                # Shift semitones to fill the high-fret "dead zone"
                atom_shifted = librosa.effects.pitch_shift(atom, sr=sr, n_steps=12)
                new_fret = fret + 12
                save_name_syn = f"str{string_num}_fr{new_fret}_SYNTH_{base_name}_{start_idx}.wav"
                sf.write(os.path.join(OUTPUT_DIR, save_name_syn), atom_shifted, sr)

if __name__ == "__main__":
    audio_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
    jams_files = sorted(glob.glob(os.path.join(JAMS_DIR, "*.jams")))
    
    if len(audio_files) == 0 or len(jams_files) == 0:
        print(f"❌ Check paths. Audio: {len(audio_files)}, JAMS: {len(jams_files)}")
        exit()

    print(f"📂 Harvesting from {len(audio_files)} Hex-Pickup files...")
    for aud, jam in tqdm(zip(audio_files, jams_files), total=len(audio_files)):
        process_file(jam, aud)
        
    print(f"✅ Harvesting Complete. Atoms saved to: {OUTPUT_DIR}")