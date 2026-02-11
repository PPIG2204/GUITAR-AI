import os
import glob
import jams
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# =========================
# CONFIG
# =========================
AUDIO_DIR = r"../../data/audio_mono-mic/"
JAMS_DIR = r"../../data/annotation/" # Ensure this matches your folder name!
OUTPUT_DIR = r"../../data/atoms/"
SAMPLE_RATE = 22050
AUGMENT_HIGH_FRETS = True 

# MIDI Base for Open Strings (Standard Tuning)
# 0=E2 (40), 1=A2 (45), 2=D3 (50), 3=G3 (55), 4=B3 (59), 5=E4 (64)
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
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"⚠️ Error loading {os.path.basename(audio_path)}: {e}")
        return

    # Iterate over ALL annotations, but process ONLY 'note_midi'
    for ann in jam.annotations:
        if ann.namespace != 'note_midi':
            continue
            
        # Extract string number from metadata (0=E2, 5=E4)
        # GuitarSet metadata usually stores it in 'data_source' as '0', '1', etc.
        try:
            string_num = int(ann.annotation_metadata.data_source)
        except:
            # Fallback/Safety: Skip if we can't identify the string
            continue

        for obs in ann.data:
            # In 'note_midi', value is the MIDI Pitch (float), e.g., 64.3
            midi_pitch = obs.value
            start_time = obs.time
            duration = obs.duration
            
            # Skip noise/short blips
            if duration < 0.1: continue
            
            # Calculate Fret (MIDI - OpenString)
            # E.g., Playing A (45) on E-string (40) -> Fret 5
            ref_midi = OPEN_STRING_MIDI[string_num]
            fret = int(round(midi_pitch - ref_midi))
            
            if fret < 0 or fret > 24: continue 

            # Slice Audio
            start_idx = int(start_time * sr)
            end_idx = int((start_time + duration) * sr)
            
            # Anti-Click
            start_idx = find_zero_crossing(y, start_idx)
            end_idx = find_zero_crossing(y, end_idx)
            
            atom = y[start_idx : end_idx]
            if len(atom) < 100: continue

            # Apply Fade
            fade_len = int(0.005 * sr)
            if len(atom) > 2 * fade_len:
                atom[:fade_len] *= np.linspace(0, 1, fade_len)
                atom[-fade_len:] *= np.linspace(1, 0, fade_len)

            # Save Base Atom
            # Naming: str{0-5}_fr{0-24}_sourceFile_id.wav
            base_name = os.path.basename(audio_path).replace(".wav", "")
            save_name = f"str{string_num}_fr{fret}_{base_name}_{start_idx}.wav"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            sf.write(save_path, atom, sr)
            
            # Synthetic High-Fret Generation
            if AUGMENT_HIGH_FRETS and fret < 8:
                # Shift +12 semitones
                atom_shifted = librosa.effects.pitch_shift(atom, sr=sr, n_steps=12)
                new_fret = fret + 12
                save_name_syn = f"str{string_num}_fr{new_fret}_SYNTH_{base_name}_{start_idx}.wav"
                sf.write(os.path.join(OUTPUT_DIR, save_name_syn), atom_shifted, sr)

if __name__ == "__main__":
    audio_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
    jams_files = sorted(glob.glob(os.path.join(JAMS_DIR, "*.jams")))
    
    # Safety Check
    if len(audio_files) == 0:
        print("❌ No Audio found! Check path:", AUDIO_DIR)
        exit()
    if len(jams_files) == 0:
        print("❌ No JAMS found! Check path:", JAMS_DIR)
        exit()

    print(f"📂 Found {len(audio_files)} pairs.")
    print(f"🔪 Slicing into {OUTPUT_DIR}...")
    
    # Use Zip but ensure sorting matches (GuitarSet filenames usually align)
    for aud, jam in tqdm(zip(audio_files, jams_files), total=len(audio_files)):
        process_file(jam, aud)
        
    print("✅ Slicing Complete. Check your folder!")