# src/1_preprocess.py
import os
import numpy as np
import librosa
import jams
from tqdm import tqdm

import config
import paths

# =========================
# GLOBAL CONSTANTS
# =========================
OPEN_STRINGS = np.array([40, 45, 50, 55, 59, 64], dtype=np.int32)
N_STRINGS = 6
MAX_FRET = 20
N_FRETS = MAX_FRET + 1
PLAYER_TEST_ID = "05"  # Player 05 is our hold-out test set

# =========================
# 1. DYNAMIC FEATURE EXTRACTION
# =========================
def extract_audio_features(y, sr):
    """Dynamically extracts CQT, Mel, or STFT based on config.py"""
    
    if config.FEATURE_TYPE == 'CQT':
        spec = librosa.cqt(
            y, sr=sr, hop_length=config.HOP_LENGTH,
            n_bins=config.CQT_BINS, bins_per_octave=config.BINS_PER_OCTAVE
        )
        spec = np.abs(spec)
        
    elif config.FEATURE_TYPE == 'MEL':
        spec = librosa.feature.melspectrogram(
            y=y, sr=sr, hop_length=config.HOP_LENGTH,
            n_fft=config.N_FFT, n_mels=config.N_MELS
        )
        
    elif config.FEATURE_TYPE == 'STFT':
        spec = librosa.stft(y, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
        spec = np.abs(spec)
        
    else:
        raise ValueError(f"Unknown FEATURE_TYPE: {config.FEATURE_TYPE}")

    # THE FIX: Log-scale with an 80dB noise floor, then Normalize [0, 1]
    spec_db = librosa.amplitude_to_db(spec, ref=np.max, top_db=80)
    spec_db = (spec_db + 80) / 80.0 
    
    # Return as (Time, Freq)
    return spec_db.T.astype(np.float32)

# =========================
# 2. JAMS PARSING
# =========================
def create_label_matrix(jam_path, num_frames, sr):
    labels = np.zeros((num_frames, N_STRINGS, N_FRETS), dtype=np.float32)
    
    try:
        jam = jams.load(str(jam_path))
    except Exception:
        return labels

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
            start_f = librosa.time_to_frames(obs.time, sr=sr, hop_length=config.HOP_LENGTH)
            dur_f   = librosa.time_to_frames(obs.duration, sr=sr, hop_length=config.HOP_LENGTH)
            end_f   = start_f + dur_f

            fret = int(round(obs.value - OPEN_STRINGS[string_idx]))
            
            if 0 <= fret <= MAX_FRET:
                s = max(0, start_f)
                e = min(num_frames, end_f)
                labels[s:e, string_idx, fret] = 1.0
                
    return labels

# =========================
# 3. REAL DATA PROCESSING
# =========================
def process_guitarset():
    audio_files = sorted([f for f in os.listdir(paths.RAW_AUDIO) if f.endswith('.wav')])
    print(f"\n🎸 Processing {len(audio_files)} GuitarSet files using {config.FEATURE_TYPE}...")

    total_chunks = 0
    
    for wav_file in tqdm(audio_files, unit="file"):
        file_id = wav_file.replace("_mic.wav", "")
        mode = "test" if file_id.startswith(PLAYER_TEST_ID) else "train"
        save_dir = paths.TEST_DATA if mode == "test" else paths.TRAIN_DATA

        audio_path = paths.RAW_AUDIO / wav_file
        jams_path  = paths.ANNOTATION / f"{file_id}.jams"

        if not jams_path.exists():
            continue

        # Extract features and labels
        y, sr = librosa.load(str(audio_path), sr=config.SAMPLE_RATE, mono=True)
        features = extract_audio_features(y, sr)
        labels = create_label_matrix(jams_path, features.shape[0], sr)

        # Slide Window Chunking
        for start_idx in range(0, features.shape[0] - config.CONTEXT_LENGTH, config.STRIDE):
            end_idx = start_idx + config.CONTEXT_LENGTH
            
            feat_chunk = features[start_idx:end_idx, :]
            label_chunk = labels[start_idx:end_idx, :, :]

            chunk_name = f"{file_id}_chunk{start_idx}.npz"
            np.savez_compressed(
                save_dir / chunk_name,
                features=feat_chunk,  # Renamed 'cqt' to 'features' to be universal
                labels=label_chunk
            )
            total_chunks += 1
            
    print(f"✅ Generated {total_chunks:,} real data chunks.")

# =========================
# 4. SYNTHETIC DATA PROCESSING
# =========================
def process_synthetic():
    if not config.USE_SYNTHETIC_DATA:
        print("⏭️ Synthetic data disabled in config. Skipping.")
        return

    if not paths.SYNTH_DIR.exists():
        print(f"⚠️ Synthetic directory {paths.SYNTH_DIR} not found. Run synth generation first.")
        return

    wav_files = sorted([f for f in os.listdir(paths.SYNTH_DIR) if f.endswith(".wav")])
    print(f"\n🧬 Processing {len(wav_files)} Synthetic Chords using {config.FEATURE_TYPE}...")

    for wav_name in tqdm(wav_files, unit="file"):
        file_id = wav_name.replace(".wav", "")
        wav_path = paths.SYNTH_DIR / wav_name
        lbl_path = paths.SYNTH_DIR / f"{file_id}_labels.npy"

        if not lbl_path.exists():
            continue

        y, sr = librosa.load(str(wav_path), sr=config.SAMPLE_RATE)
        features = extract_audio_features(y, sr)

        # Synthetics are exactly 1 second, shape labels to match feature length
        raw_labels = np.load(str(lbl_path))
        full_labels = np.repeat(raw_labels[np.newaxis, :, :], config.CONTEXT_LENGTH, axis=0)

        np.savez_compressed(
            paths.SYNTH_DATA / f"{file_id}.npz",
            features=features[:config.CONTEXT_LENGTH],
            labels=full_labels[:config.CONTEXT_LENGTH]
        )
        
    print("✅ Synthetic data processed.")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print(f"==================================================")
    print(f"🚀 INITIALIZING EXPERIMENT: {config.EXPERIMENT_NAME}")
    print(f"==================================================")
    
    process_guitarset()
    process_synthetic()
    
    print("\n🎉 Preprocessing Pipeline Complete!")
    print(f"📂 Output routed to: {paths.EXP_DIR}")