import os
import argparse
import numpy as np
import librosa
import torch
import scipy.ndimage as ndimage
from scipy.signal import medfilt
import pretty_midi

# Import your model definition
# Ensure model.py is in the same folder or adjusted path
from model import GuitarTranscriberCNN

# =========================
# CONFIG
# =========================
MODEL_PATH = "./saved_models/guitar_model.pth"
SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_BINS = 192
BINS_PER_OCTAVE = 24
CONTEXT = 128
THRESHOLD = 0.65  # Confidence threshold

STRING_NAMES = ["e", "B", "G", "D", "A", "E"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 1. AUDIO PROCESSOR
# =========================
def compute_cqt(audio_path):
    print(f"Loading audio: {audio_path}")
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return None

    # Trim silence
    y, _ = librosa.effects.trim(y)

    cqt = librosa.cqt(
        y,
        sr=sr,
        hop_length=HOP_LENGTH,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE
    )
    
    cqt = np.abs(cqt)
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)

    # NORMALIZE
    cqt_db -= cqt_db.min()
    cqt_db /= (cqt_db.max() + 1e-8)
    
    return cqt_db.T.astype(np.float32)

# =========================
# 2. INFERENCE ENGINE
# =========================
def run_inference(model, cqt):
    T = cqt.shape[0]
    if T < CONTEXT:
        padding = np.zeros((CONTEXT - T, cqt.shape[1]))
        cqt = np.vstack([cqt, padding])
        T = CONTEXT

    preds = []
    step = CONTEXT // 2 
    
    with torch.no_grad():
        for i in range(0, T - CONTEXT, step):
            chunk = cqt[i : i+CONTEXT]
            
            # Reshape for model (Batch, Channel, Freq, Time)
            # Transpose chunk so Freq is axis 2
            x = torch.from_numpy(chunk.T).float().unsqueeze(0).unsqueeze(0).to(device)
            
            y = torch.sigmoid(model(x))
            preds.append(y.cpu().numpy().squeeze()) 
    
    # Reconstruct
    full_pred = np.zeros((T, 6, 21))
    count_map = np.zeros((T, 6, 21))
    
    for idx, i in enumerate(range(0, T - CONTEXT, step)):
        y = preds[idx]
        full_pred[i : i+CONTEXT] += y
        count_map[i : i+CONTEXT] += 1
        
    count_map[count_map == 0] = 1
    full_pred /= count_map
    
    return full_pred

# =========================
# 3. DSP & LOGIC TOOLS
# =========================
def apply_temporal_smoothing(binary_matrix, kernel_size=5):
    """
    Applies Median Filter along time axis to remove jitter.
    """
    return medfilt(binary_matrix, kernel_size=(kernel_size, 1, 1))

def clean_duplicates(frame):
    """
    Pick best string/fret for each pitch. Prefer lower frets.
    """
    string_offsets = [64, 59, 55, 50, 45, 40] # e B G D A E
    
    candidates = []
    for s in range(6):
        fret = np.argmax(frame[s])
        prob = frame[s, fret]
        
        # Note: since input is binary, prob is always 1.0 here, 
        # but we keep logic for robustness.
        if fret > 0 and prob > 0.5:
            pitch = string_offsets[s] + fret
            candidates.append({
                's': s, 'fret': fret, 'pitch': pitch, 'prob': prob
            })
            
    if not candidates:
        return np.zeros_like(frame)

    # Filter: Keep only lowest fret for each pitch
    best_candidates = {}
    for c in candidates:
        p = c['pitch']
        if p not in best_candidates:
            best_candidates[p] = c
        else:
            if c['fret'] < best_candidates[p]['fret']:
                best_candidates[p] = c

    clean_frame = np.zeros_like(frame)
    for p, c in best_candidates.items():
        clean_frame[c['s'], c['fret']] = c['prob']
        
    return clean_frame

def batch_clean_duplicates(binary_matrix):
    T = binary_matrix.shape[0]
    cleaned = np.zeros_like(binary_matrix)
    for t in range(T):
        cleaned[t] = clean_duplicates(binary_matrix[t])
    return cleaned

# =========================
# 4. OUTPUT FORMATTERS
# =========================
def matrix_to_tab(binary):
    DOWNSAMPLE = 4 
    BLOCK_SIZE = 80
    T = binary.shape[0]
    full_tab = {i: [] for i in range(6)}
    last_fret = [-1] * 6 
    
    for t in range(0, T, DOWNSAMPLE):
        frame = binary[t] # Already cleaned
        for s in range(6):
            visual_idx = 5 - s 
            fret_idx = np.argmax(frame[s])
            
            if frame[s, fret_idx] > 0.5 and fret_idx > 0:
                if fret_idx != last_fret[s]:
                    char = str(fret_idx)
                    last_fret[s] = fret_idx
                else:
                    char = "-" 
            else:
                char = "-"
                last_fret[s] = -1
            
            if len(char) == 1: 
                full_tab[visual_idx].append(char + "-")
            else:
                full_tab[visual_idx].append(char)

    output_str = ""
    total_len = len(full_tab[0])
    for i in range(0, total_len, BLOCK_SIZE):
        output_str += "\n"
        for s in range(6):
            chunk = "".join(full_tab[s][i : min(i+BLOCK_SIZE, total_len)])
            output_str += f"{STRING_NAMES[s]}|{chunk}|\n"
    return output_str

def save_to_midi(binary, output_filename):
    midi = pretty_midi.PrettyMIDI()
    # Use Electric Guitar (Clean) -> Program 27
    inst = pretty_midi.Instrument(program=27)

    TIME_PER_FRAME = HOP_LENGTH / SAMPLE_RATE
    MIN_NOTE_DURATION = 0.10
    MAX_GAP_TO_FILL = 0.15
    STRING_OFFSETS = [64, 59, 55, 50, 45, 40] 

    T = binary.shape[0]
    raw_notes = []

    # 1. Extract Events
    for s in range(6):
        active_start = None
        active_pitch = None
        
        for t in range(T):
            frame = binary[t]
            fret = np.argmax(frame[s])
            is_playing = (frame[s, fret] > 0.5)
            pitch = STRING_OFFSETS[s] + fret
            
            if active_start is not None:
                if not is_playing or (is_playing and pitch != active_pitch):
                    raw_notes.append({
                        'pitch': active_pitch,
                        'start': active_start * TIME_PER_FRAME,
                        'end': t * TIME_PER_FRAME
                    })
                    active_start = None
            
            if is_playing and active_start is None:
                active_start = t
                active_pitch = pitch
    
    # 2. Smooth & Merge
    raw_notes.sort(key=lambda x: x['start'])
    from collections import defaultdict
    notes_by_pitch = defaultdict(list)
    for n in raw_notes: notes_by_pitch[n['pitch']].append(n)
        
    for pitch, group in notes_by_pitch.items():
        group.sort(key=lambda x: x['start'])
        if not group: continue
        
        current = group[0]
        merged = []
        
        for next_note in group[1:]:
            gap = next_note['start'] - current['end']
            if gap < MAX_GAP_TO_FILL:
                current['end'] = max(current['end'], next_note['end'])
            else:
                merged.append(current)
                current = next_note
        merged.append(current)
        
        for m in merged:
            if (m['end'] - m['start']) >= MIN_NOTE_DURATION:
                note = pretty_midi.Note(
                    velocity=100, pitch=int(m['pitch']),
                    start=m['start'], end=m['end']
                )
                inst.notes.append(note)

    midi.instruments.append(inst)
    midi.write(output_filename)
    print(f"🎹 MIDI saved to: {output_filename}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Guitar CNN on YouTube Audio")
    parser.add_argument("audio_file", type=str, help="Path to audio file")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found! Train it first.")
        exit()

    # 1. Setup
    print("⏳ Loading model...")
    # weights_only=True is safer for PyTorch 2.6+
    try:
        model = GuitarTranscriberCNN().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    except:
        # Fallback for older torch versions
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
    model.eval()

    # 2. Inference
    cqt = compute_cqt(args.audio_file)
    if cqt is None: exit()
    
    print("🧠 Thinking...")
    preds = run_inference(model, cqt)
    
    # 3. PIPELINE: Threshold -> Smooth -> Morph -> Logic
    # 3.1 Threshold
    binary = (preds > THRESHOLD).astype(float)
    
    # 3.2 Signal Smoothing (Median Filter - Fixes Jitter)
    print("🪄 Smoothing temporal jitter...")
    binary = apply_temporal_smoothing(binary, kernel_size=5)
    
    # 3.3 Morphology (Close small gaps)
    print("🧽 Filling signal gaps...")
    binary = ndimage.binary_closing(binary, structure=np.ones((4,1,1)))

    # 3.4 Musical Logic (Fix unisons / Impossible fingerings)
    print("🧹 Cleaning up duplicates...")
    binary = batch_clean_duplicates(binary) 

    # 4. Output
    print("\n" + "="*40)
    print(f"🎸 TABLATURE FOR: {os.path.basename(args.audio_file)}")
    print("="*40)
    print(matrix_to_tab(binary))
    print("="*40)

    midi_filename = os.path.splitext(args.audio_file)[0] + ".mid"
    save_to_midi(binary, midi_filename)