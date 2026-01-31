import os
import numpy as np
import librosa
import jams
from tqdm import tqdm

# =========================
# CONFIG
# =========================
MODES = ["train", "test"]

SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_BINS = 192
BINS_PER_OCTAVE = 24

OPEN_STRINGS = np.array([40, 45, 50, 55, 59, 64], dtype=np.int32)
N_STRINGS = 6
MAX_FRET = 20
N_FRETS = MAX_FRET + 1

# --- THE FIX: SLIDING WINDOW CONFIG ---
CONTEXT_LENGTH = 128   # The window size the model sees (approx 3 seconds)
STRIDE = 64            # 50% Overlap. This fixes the "Flickering/Continuity" issue.
# --------------------------------------

PRINT_EVERY = 50 

# =========================
# CORE
# =========================
def process_folder(mode: str):
    audio_dir = f"./data/{mode}/audio/"
    jams_dir  = f"./data/{mode}/jams/"
    save_dir  = f"./processed_data/{mode}/"
    os.makedirs(save_dir, exist_ok=True)

    jam_files = sorted(f for f in os.listdir(jams_dir) if f.endswith(".jams"))
    print(f"\n=== [{mode.upper()}] PREPROCESSING | {len(jam_files)} files ===")

    # ---- global stats ----
    total_chunks = 0
    total_polyphonic_frames = 0  # To prove to Leader that chords exist
    string_activity = np.zeros(N_STRINGS, dtype=np.int64)
    fret_activity   = np.zeros(N_FRETS, dtype=np.int64)

    for idx, jam_file in enumerate(tqdm(
        jam_files,
        desc=f"{mode.upper()}",
        unit="file",
        dynamic_ncols=True
    )):
        file_id = jam_file[:-5]
        wav_file = f"{file_id}_mic.wav"

        audio_path = os.path.join(audio_dir, wav_file)
        jams_path  = os.path.join(jams_dir, jam_file)

        if not os.path.exists(audio_path):
            continue

        # =========================
        # 1. AUDIO → CQT
        # =========================
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        cqt = librosa.cqt(
            y,
            sr=sr,
            hop_length=HOP_LENGTH,
            n_bins=N_BINS,
            bins_per_octave=BINS_PER_OCTAVE
        )

        cqt = np.abs(cqt)
        cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)

        # Normalize [0,1]
        cqt_db -= cqt_db.min()
        cqt_db /= (cqt_db.max() + 1e-8)
        cqt_db = cqt_db.T.astype(np.float32)  # Shape: (Total_Time, Freq_Bins)

        num_frames_total = cqt_db.shape[0]

        # =========================
        # 2. LABEL MATRIX (POLYPHONIC)
        # =========================
        # Shape: (Total_Time, 6 strings, 21 frets)
        full_labels = np.zeros(
            (num_frames_total, N_STRINGS, N_FRETS),
            dtype=np.float32
        )

        try:
            jam = jams.load(jams_path)
        except Exception:
            continue

        # Parse JAMS
        for ann in jam.annotations:
            if ann.namespace != "note_midi":
                continue

            try:
                # GuitarSet data_source is usually "0", "1", ... "5"
                string_idx = int(ann.annotation_metadata.data_source)
            except Exception:
                continue

            if not (0 <= string_idx < N_STRINGS):
                continue

            for obs in ann:
                start_f = librosa.time_to_frames(obs.time, sr=sr, hop_length=HOP_LENGTH)
                dur_f   = librosa.time_to_frames(obs.duration, sr=sr, hop_length=HOP_LENGTH)
                end_f   = start_f + dur_f

                # Calculate fret based on pitch
                fret = int(round(obs.value - OPEN_STRINGS[string_idx]))
                
                if 0 <= fret <= MAX_FRET:
                    s = max(0, start_f)
                    e = min(num_frames_total, end_f)
                    full_labels[s:e, string_idx, fret] = 1.0

        # =========================
        # 3. SLIDING WINDOW CHUNKING (The Fix)
        # =========================
        # We slice the long track into small overlapping pieces
        
        for start_idx in range(0, num_frames_total - CONTEXT_LENGTH, STRIDE):
            end_idx = start_idx + CONTEXT_LENGTH
            
            # Slice Input and Output
            cqt_chunk = cqt_db[start_idx:end_idx, :]       # (128, 192)
            label_chunk = full_labels[start_idx:end_idx, :, :] # (128, 6, 21)

            # Check if chunk has any info (optional, but saves space)
            if np.sum(label_chunk) == 0:
                # Skip pure silence chunks to balance data? 
                # For now, let's keep them to learn silence.
                pass

            # Save as separate mini-file
            chunk_name = f"{file_id}_chunk{start_idx}.npz"
            save_path = os.path.join(save_dir, chunk_name)
            
            np.savez_compressed(
                save_path,
                cqt=cqt_chunk,
                labels=label_chunk
            )
            
            # Stats Collection
            total_chunks += 1
            
            # Count how many frames in this chunk have >1 note playing (Polyphony check)
            # Sum over strings (axis 1), if > 1, it's a chord
            active_notes_per_frame = label_chunk.sum(axis=(1, 2)) # Shape (128,)
            poly_frames_in_chunk = (active_notes_per_frame > 1).sum()
            total_polyphonic_frames += poly_frames_in_chunk
            
            string_activity += label_chunk.sum(axis=(0, 2)).astype(np.int64)
            fret_activity   += label_chunk.sum(axis=(0, 1)).astype(np.int64)

    # =========================
    # FINAL SUMMARY
    # =========================
    print(f"\n--- [{mode.upper()}] SUMMARY ---")
    print(f"Total Chunks Generated : {total_chunks:,}")
    print(f"Total Polyphonic Frames: {total_polyphonic_frames:,}")
    print(f"   (Evidence that GuitarSet is NOT monophonic)")

    print("\nString activity:")
    for i, v in enumerate(string_activity):
        print(f"  String {i}: {v:,}")

# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    for mode in MODES:
        process_folder(mode)

    print("\n✅ Preprocessing finished cleanly with Overlap.\n")