import os
import numpy as np
from tqdm import tqdm

# =========================
# CONFIG
# =========================
SOURCE_DIR = "./processed_data/train/"
AUGMENT_SHIFTS = [-2, -1, 1, 2]  # Shift -2 semitones to +2 semitones
# This creates 4 new copies for every 1 original file (5x Data!)

N_FRETS = 21
N_BINS = 192
BINS_PER_OCTAVE = 24

def augment_file(file_path):
    try:
        data = np.load(file_path)
        cqt = data["cqt"]       # (T, 192)
        labels = data["labels"] # (T, 6, 21)
    except Exception:
        return

    basename = os.path.splitext(os.path.basename(file_path))[0]
    folder = os.path.dirname(file_path)

    for shift in AUGMENT_SHIFTS:
        # 1. SHIFT CQT (The "Image Roll")
        # Shift frequency bins up/down
        # CQT is (Time, Freq), so we roll axis 1
        
        # Calculate bin shift (24 bins per octave / 12 semitones = 2 bins per semitone)
        # Check your CQT params. If bins_per_octave=24, then 1 semitone = 2 bins.
        bins_shift = shift * (BINS_PER_OCTAVE // 12)
        
        cqt_aug = np.roll(cqt, bins_shift, axis=1)
        
        # Handle edges (Zero out the rolled-over part so low notes don't appear at high freq)
        if bins_shift > 0:
            cqt_aug[:, :bins_shift] = 0
        elif bins_shift < 0:
            cqt_aug[:, bins_shift:] = 0

        # 2. SHIFT LABELS (The Logic)
        # Label shape: (Time, String, Fret)
        # We roll the Fret axis (axis 2)
        labels_aug = np.zeros_like(labels)
        
        # We iterate strings to be safe
        for s in range(6):
            # Roll the fret probabilities
            # shift is directly semitones/frets
            original_string_data = labels[:, s, :]
            shifted_string_data = np.roll(original_string_data, shift, axis=1)
            
            # Mask out invalid frets (wrapping around)
            if shift > 0:
                shifted_string_data[:, :shift] = 0
            elif shift < 0:
                shifted_string_data[:, shift:] = 0
                
            labels_aug[:, s, :] = shifted_string_data

        # 3. SAVE FRANKENSTEIN FILE
        new_filename = f"{basename}_aug{shift:+d}.npz"
        save_path = os.path.join(folder, new_filename)
        
        np.savez_compressed(
            save_path,
            cqt=cqt_aug,
            labels=labels_aug
        )

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    files = [
        os.path.join(SOURCE_DIR, f) 
        for f in os.listdir(SOURCE_DIR) 
        if f.endswith(".npz") and "_aug" not in f # Don't re-augment augmented files
    ]
    
    print(f"🧬 Creating Frankenstein Data from {len(files)} files...")
    print(f"   Shifts: {AUGMENT_SHIFTS}")
    
    for f in tqdm(files, unit="file"):
        augment_file(f)
        
    print("\n✅ Augmentation Complete.")
    print(f"   Dataset size increased by {len(AUGMENT_SHIFTS) + 1}x")