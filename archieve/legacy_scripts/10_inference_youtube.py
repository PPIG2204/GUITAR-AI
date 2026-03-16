import os
import sys
import argparse
import numpy as np
import librosa
import torch
import pretty_midi

# 0. PATH INJECTION
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paths import MODEL_DIR
from model import GuitarTranscriberCNN

# ==================================================
# VITERBI DECODER
# ==================================================
class GuitarViterbi:
    def __init__(self, n_frets=21, threshold=0.65):
        self.n_frets = n_frets
        self.n_states = n_frets + 1  
        self.silence_idx = n_frets
        self.threshold = threshold 
        
        self.jump_penalty = 0.5    
        self.onset_penalty = 0.2   
        self.sustain_bonus = 0.1   
        
    def decode_string(self, prob_matrix_1d):
        T, F = prob_matrix_1d.shape
        silence_scores = np.full((T, 1), self.threshold)
        emissions = np.hstack([prob_matrix_1d, silence_scores])
        eps = 1e-6 
        log_emissions = np.log(emissions + eps)
        
        path_probs = np.zeros((T, self.n_states))
        path_pointers = np.zeros((T, self.n_states), dtype=int)
        path_probs[0] = log_emissions[0]
        
        trans_mat = np.zeros((self.n_states, self.n_states))
        fret_indices = np.arange(self.n_frets)
        for f_prev in range(self.n_frets):
            dist = np.abs(fret_indices - f_prev)
            trans_mat[f_prev, :self.n_frets] = - (dist * self.jump_penalty)
            trans_mat[f_prev, f_prev] += self.sustain_bonus 

        trans_mat[self.silence_idx, self.silence_idx] = 0        
        trans_mat[:self.n_frets, self.silence_idx]    = 0        
        trans_mat[self.silence_idx, :self.n_frets]    = -self.onset_penalty 

        for t in range(1, T):
            scores = path_probs[t-1][:, None] + trans_mat
            path_probs[t] = np.max(scores, axis=0) + log_emissions[t]
            path_pointers[t] = np.argmax(scores, axis=0)
            
        best_path = np.zeros(T, dtype=int)
        best_path[-1] = np.argmax(path_probs[-1])
        for t in range(T-2, -1, -1):
            best_path[t] = path_pointers[t+1, best_path[t+1]]
            
        return best_path

# =========================
# CONFIG
# =========================
MODEL_PATH = str(MODEL_DIR / "guitar_model.pth")
SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_BINS = 192
BINS_PER_OCTAVE = 24
CONTEXT = 128
THRESHOLD = 0.65 

STRING_NAMES = ["e", "B", "G", "D", "A", "E"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 1. AUDIO PROCESSOR
# =========================
def compute_cqt(audio_path):
    print(f"📂 Loading audio: {audio_path}")
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return None

    y, _ = librosa.effects.trim(y)
    cqt = librosa.cqt(y, sr=sr, hop_length=HOP_LENGTH, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE)
    cqt = np.abs(cqt)
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
    cqt_db -= cqt_db.min()
    cqt_db /= (cqt_db.max() + 1e-8)
    return cqt_db.T.astype(np.float32)

# =========================
# 2. INFERENCE ENGINE (FINALLY FIXED)
# =========================
def run_inference(model, cqt):
    T = cqt.shape[0]
    n_chunks = T // CONTEXT
    if n_chunks == 0: return None

    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for i in range(n_chunks):
            chunk = cqt[i*CONTEXT : (i+1)*CONTEXT].T 
            x = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.amp.autocast('cuda'):
                y = torch.sigmoid(model(x)) 
            
            # y is shape (Batch, 128, 6, 21)
            # Squeeze removes the batch, leaving exactly (128, 6, 21)
            y_np = y.cpu().numpy().squeeze() 
            
            # No reshaping, no repeating. Just append the 128 frames.
            all_preds.append(y_np)

    # Concatenate time windows -> Final shape: (Total_Time, 6, 21)
    return np.concatenate(all_preds, axis=0)

# =========================
# 3. TAB & MIDI FORMATTERS
# =========================
def matrix_to_tab(binary):
    DOWNSAMPLE = 2 
    BLOCK_SIZE = 100
    T = binary.shape[0]
    full_tab = {i: [] for i in range(6)}
    last_fret = [-1] * 6 
    
    for t in range(0, T, DOWNSAMPLE):
        frame = binary[t]
        for s in range(6):
            visual_idx = 5 - s 
            fret_idx = np.argmax(frame[s])
            
            if frame[s, fret_idx] > 0.5:
                if fret_idx != last_fret[s]:
                    char = str(fret_idx)
                    last_fret[s] = fret_idx
                else: 
                    char = "-" 
            else:
                char = "-"
                last_fret[s] = -1
                
            full_tab[visual_idx].append(char + "-" if len(char) == 1 else char)

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
    inst = pretty_midi.Instrument(program=24) 
    TIME_PER_FRAME = HOP_LENGTH / SAMPLE_RATE
    STRING_OFFSETS = [64, 59, 55, 50, 45, 40] 

    for s in range(6):
        active_start, active_pitch = None, None
        for t in range(binary.shape[0]):
            frame = binary[t]
            fret = np.argmax(frame[s])
            is_playing = (frame[s, fret] > 0.5)
            pitch = STRING_OFFSETS[s] + fret
            if active_start is not None:
                if not is_playing or pitch != active_pitch:
                    inst.notes.append(pretty_midi.Note(100, int(active_pitch), active_start * TIME_PER_FRAME, t * TIME_PER_FRAME))
                    active_start = None
            if is_playing and active_start is None:
                active_start, active_pitch = t, pitch

    midi.instruments.append(inst)
    midi.write(output_filename)
    print(f"🎹 MIDI saved to: {output_filename}")

# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    print("🚀 Script started. Parsing arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", type=str, help="Path to audio file")
    args = parser.parse_args()

    print("🔌 Waking up PyTorch and GPU...")
    model = GuitarTranscriberCNN().to(device)
    
    print(f"📦 Loading model weights from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    print("✅ Model loaded! Moving to audio processing...")
    cqt = compute_cqt(args.audio_file)
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", type=str, help="Path to audio file")
    args = parser.parse_args()

    model = GuitarTranscriberCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    cqt = compute_cqt(args.audio_file)
    if cqt is not None:
        print("🧠 CNN is thinking...")
        preds = run_inference(model, cqt)
        
        if preds is not None:
            print("🪄 Applying Viterbi Physics Constraints...")
            decoder = GuitarViterbi(threshold=THRESHOLD)
            
            binary = np.zeros_like(preds)
            
            for s in range(6):
                # preds[:, s, :] is inherently (Total_Time, 21)
                path = decoder.decode_string(preds[:, s, :])
                for t, state in enumerate(path):
                    if state < 21: 
                        binary[t, s, state] = 1.0
            
            print("\n" + "="*40)
            print(f"🎸 TABLATURE: {os.path.basename(args.audio_file)}")
            print("="*40)
            print(matrix_to_tab(binary))
            
            midi_out = os.path.splitext(args.audio_file)[0] + ".mid"
            save_to_midi(binary, midi_out)
            print("="*40)
            print("✅ Process Complete.")