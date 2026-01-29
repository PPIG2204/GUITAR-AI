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

PRINT_EVERY = 50  # console stats frequency

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
    total_frames = 0
    active_frames = 0
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
        save_path  = os.path.join(save_dir, f"{file_id}.npz")

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

        # normalize [0,1]
        cqt_db -= cqt_db.min()
        cqt_db /= (cqt_db.max() + 1e-8)

        cqt_db = cqt_db.T.astype(np.float32)  # (T, F)
        num_frames = cqt_db.shape[0]
        total_frames += num_frames

        # =========================
        # 2. LABEL MATRIX
        # =========================
        labels = np.zeros(
            (num_frames, N_STRINGS, N_FRETS),
            dtype=np.float32
        )

        try:
            jam = jams.load(jams_path)
        except Exception:
            continue

        for ann in jam.annotations:
            if ann.namespace != "note_midi":
                continue

            try:
                string_idx = int(ann.annotation_metadata.data_source)
            except Exception:
                continue

            if not (0 <= string_idx < N_STRINGS):
                continue

            for obs in ann:
                start = librosa.time_to_frames(
                    obs.time, sr=sr, hop_length=HOP_LENGTH
                )
                end = start + librosa.time_to_frames(
                    obs.duration, sr=sr, hop_length=HOP_LENGTH
                )

                fret = int(round(obs.value - OPEN_STRINGS[string_idx]))
                if not (0 <= fret <= MAX_FRET):
                    continue

                s = max(0, start)
                e = min(num_frames, end)
                labels[s:e, string_idx, fret] = 1.0

        # =========================
        # 3. STATS
        # =========================
        frame_active = labels.sum(axis=(1, 2)) > 0
        active_frames += frame_active.sum()

        string_activity += labels.sum(axis=(0, 2)).astype(np.int64)
        fret_activity   += labels.sum(axis=(0, 1)).astype(np.int64)

        # =========================
        # 4. SAVE
        # =========================
        np.savez_compressed(
            save_path,
            cqt=cqt_db,
            labels=labels
        )

        # =========================
        # 5. CONSOLE LOG
        # =========================
        if (idx + 1) % PRINT_EVERY == 0:
            print(
                f"[{mode}] {idx+1:3d}/{len(jam_files)} | "
                f"frames: {total_frames:,} | "
                f"active: {active_frames/total_frames:.2%}"
            )

    # =========================
    # FINAL SUMMARY
    # =========================
    print(f"\n--- [{mode.upper()}] SUMMARY ---")
    print(f"Total frames      : {total_frames:,}")
    print(f"Active frames     : {active_frames:,} "
          f"({active_frames/total_frames:.2%})")

    print("\nString activity (frame-count):")
    for i, v in enumerate(string_activity):
        print(f"  String {i}: {v:,}")

    print("\nTop frets (most active):")
    for f in np.argsort(fret_activity)[-5:][::-1]:
        print(f"  Fret {f:2d}: {fret_activity[f]:,}")

# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    for mode in MODES:
        process_folder(mode)

    print("\n✅ Preprocessing finished cleanly.\n")
