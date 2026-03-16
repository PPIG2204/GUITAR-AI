import os
import numpy as np
import librosa
from tqdm import tqdm
import sys

# Path injection to find paths.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paths import SYNTH_DIR, BASE_DIR

# =========================
# CONFIG
# =========================
SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_BINS = 192
BINS_PER_OCTAVE = 24
CONTEXT_LENGTH = 128  # Matches model input

SAVE_DIR = BASE_DIR / "processed_data" / "synthetic"
os.makedirs(SAVE_DIR, exist_ok=True)

def process_synthetic_chords():
    # Find all generated .wav and .npy pairs
    wav_files = sorted([f for f in os.listdir(SYNTH_DIR) if f.endswith(".wav")])
    
    print(f"🧪 Preprocessing {len(wav_files)} Synthetic Chords...")

    for wav_name in tqdm(wav_files):
        file_id = wav_name.replace(".wav", "")
        wav_path = os.path.join(SYNTH_DIR, wav_name)
        lbl_path = os.path.join(SYNTH_DIR, f"{file_id}_labels.npy")

        if not os.path.exists(lbl_path):
            continue

        # 1. AUDIO -> CQT
        y, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
        
        cqt = librosa.cqt(
            y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, 
            n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE
        )
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        
        # Normalize and Transpose (Time, Freq)
        cqt_db -= cqt_db.min()
        cqt_db /= (cqt_db.max() + 1e-8)
        cqt_feat = cqt_db.T.astype(np.float32)

        # 2. LOAD LABELS
        # Shape: (6, 21) -> Expand to (Time, 6, 21)
        raw_labels = np.load(lbl_path)
        full_labels = np.repeat(raw_labels[np.newaxis, :, :], CONTEXT_LENGTH, axis=0)

        # 3. SAVE AS NPZ
        # Ensure we only save the exact context length the model expects
        np.savez_compressed(
            os.path.join(SAVE_DIR, f"{file_id}.npz"),
            cqt=cqt_feat[:CONTEXT_LENGTH],
            labels=full_labels[:CONTEXT_LENGTH]
        )

if __name__ == "__main__":
    process_synthetic_chords()
    print(f"✅ Preprocessing complete. Data ready in: {SAVE_DIR}")