import sys
import os
import librosa
import numpy as np
from tqdm import tqdm
import shutil

# Add the 'src' directory to the system path to find paths.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paths import ATOM_DIR, DATA_DIR

def validate_data():
    # Define a trash folder for the "Garbage"
    TRASH_DIR = os.path.join(DATA_DIR, "trash_atoms")
    os.makedirs(TRASH_DIR, exist_ok=True)

    files = [os.path.join(ATOM_DIR, f) for f in os.listdir(ATOM_DIR) if f.endswith(".wav")]
    garbage_count = 0
    
    print(f"🔍 Scanning {len(files)} atoms for quality...")

    for f in tqdm(files, desc="Validating"):
        try:
            y, sr = librosa.load(f, sr=None)
            
            # 1. Check for Silence/Low Energy
            rms = np.sqrt(np.mean(y**2))
            if rms < 0.005:
                shutil.move(f, os.path.join(TRASH_DIR, os.path.basename(f)))
                garbage_count += 1
                continue

            # 2. Check for "Double Onsets" (The two-note problem)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            if len(onsets) > 1:
                shutil.move(f, os.path.join(TRASH_DIR, os.path.basename(f)))
                garbage_count += 1
                continue
        except Exception:
            # Move corrupted files to trash as well
            shutil.move(f, os.path.join(TRASH_DIR, os.path.basename(f)))
            garbage_count += 1

    print(f"\n✅ Validation Complete.")
    print(f"🗑️ Moved {garbage_count} suspicious files to: {TRASH_DIR}")
    print(f"🎸 Remaining clean atoms: {len(files) - garbage_count}")

if __name__ == "__main__":
    validate_data()