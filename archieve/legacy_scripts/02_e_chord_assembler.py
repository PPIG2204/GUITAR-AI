import os
import glob
import random
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# =========================
# CONFIG
# =========================
ATOM_DIR = "../../data/atoms/"
OUTPUT_DIR = "../../data/synthetic_chords/"
SAMPLE_RATE = 22050
NUM_CHORDS = 2000  # Adjust based on your GTX 1060 limits
MAX_NOTES_PER_CHORD = 4 # Common for guitar voicing

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_atom_metadata(filename):
    """Extracts string and fret from the filename format used in 02_d_note_slicing."""
    parts = filename.split('_')
    string_idx = int(parts[0].replace('str', ''))
    fret_idx = int(parts[1].replace('fr', ''))
    return string_idx, fret_idx

def assemble_chords():
    # 1. Catalog all available atoms by string
    all_atoms = glob.glob(os.path.join(ATOM_DIR, "*.wav"))
    atoms_by_string = {i: [] for i in range(6)}
    
    for f in all_atoms:
        s, _ = parse_atom_metadata(os.path.basename(f))
        atoms_by_string[s].append(f)

    print(f"🧬 Assembling {NUM_CHORDS} Frankenstein Chords...")

    for i in tqdm(range(NUM_CHORDS)):
        # 2. Randomly select 2-4 strings to participate
        num_notes = random.randint(2, MAX_NOTES_PER_CHORD)
        active_strings = random.sample(range(6), num_notes)
        
        chord_audio = np.zeros(int(SAMPLE_RATE * 1.0)) # 1-second chunks
        chord_labels = np.zeros((6, 21)) # Label for this "frame"
        
        for s in active_strings:
            if not atoms_by_string[s]: continue
            
            atom_path = random.choice(atoms_by_string[s])
            _, fret = parse_atom_metadata(os.path.basename(atom_path))
            
            # Load and mix
            y, _ = librosa.load(atom_path, sr=SAMPLE_RATE)
            
            # Pad or trim to 1 second
            if len(y) > len(chord_audio):
                y = y[:len(chord_audio)]
            
            # Mix with random slight gain variation for realism
            chord_audio[:len(y)] += y * random.uniform(0.7, 1.0)
            chord_labels[s, fret] = 1.0

        # 3. Save audio and a matching label file
        save_name = f"chord_{i:04d}.wav"
        label_name = f"chord_{i:04d}_labels.npy"
        
        sf.write(os.path.join(OUTPUT_DIR, save_name), chord_audio, SAMPLE_RATE)
        np.save(os.path.join(OUTPUT_DIR, label_name), chord_labels)

if __name__ == "__main__":
    assemble_chords()
    print(f"✅ Success. Synthetic chords saved to {OUTPUT_DIR}")