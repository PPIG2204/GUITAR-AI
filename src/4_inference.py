# src/4_inference.py
import os
import argparse
import numpy as np
import torch
import librosa

import config
import paths
from model import GuitarTranscriberCNN

# =========================
# THE PHYSICS ENGINE (Self-Contained)
# =========================
class GuitarViterbi:
    def __init__(self, n_frets=21, threshold=0.85):
        self.n_frets = n_frets
        self.n_states = n_frets + 1 
        self.silence_idx = n_frets
        self.threshold = threshold 
        self.jump_penalty = 0.5    
        self.onset_penalty = 0.2   
        self.sustain_bonus = 0.1   
        
    def decode_string(self, prob_matrix_1d):
        T = prob_matrix_1d.shape[0]
        silence_scores = np.full((T, 1), self.threshold)
        emissions = np.hstack([prob_matrix_1d, silence_scores])
        log_emissions = np.log(emissions + 1e-6)
        
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

def matrix_to_tab(binary_matrix, title):
    STRING_NAMES = ["e", "B", "G", "D", "A", "E"]
    tab_lines = {s: [f"{STRING_NAMES[s]}|"] for s in range(6)}
    
    for t in range(0, binary_matrix.shape[0], 8):
        for s in range(6):
            visual = 5 - s
            fret_vec = binary_matrix[t, s]
            fret = fret_vec.argmax()
            if fret_vec[fret] > 0.5 and fret > 0:
                tab_lines[visual].append(str(fret).ljust(2, "-"))
            else:
                tab_lines[visual].append("--")
                
    return f"\n--- {title} ---\n" + "\n".join("".join(tab_lines[s]) for s in range(6)) + "\n"

# =========================
# 1. AUDIO PROCESSOR
# =========================
def extract_features(audio_path):
    print(f"🎧 Loading audio: {audio_path}")
    try:
        y, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return None

    y, _ = librosa.effects.trim(y)
    print(f"🔄 Converting to {config.FEATURE_TYPE} Spectrogram...")

    if config.FEATURE_TYPE == 'CQT':
        spec = librosa.cqt(y, sr=sr, hop_length=config.HOP_LENGTH, n_bins=config.CQT_BINS, bins_per_octave=config.BINS_PER_OCTAVE)
        spec = np.abs(spec)
    elif config.FEATURE_TYPE == 'MEL':
        spec = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=config.HOP_LENGTH, n_fft=config.N_FFT, n_mels=config.N_MELS)
    elif config.FEATURE_TYPE == 'STFT':
        spec = librosa.stft(y, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
        spec = np.abs(spec)

    spec_db = librosa.amplitude_to_db(spec, ref=np.max)
    spec_db -= spec_db.min()
    spec_db /= (spec_db.max() + 1e-8)
    
    return spec_db.T.astype(np.float32)

# =========================
# 2. THE AI BRAIN
# =========================
def predict(model, features, device):
    print("🧠 AI is analyzing the fretboard...")
    T = features.shape[0]
    n_chunks = T // config.CONTEXT_LENGTH
    if n_chunks == 0:
        print("⚠️ Audio is too short for the context window.")
        return None

    model.eval()
    all_preds = []

    with torch.no_grad():
        for i in range(n_chunks):
            chunk = features[i*config.CONTEXT_LENGTH : (i+1)*config.CONTEXT_LENGTH].T 
            x = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0).to(device)
            
            y = torch.sigmoid(model(x))
            y_np = y.cpu().numpy().squeeze()
            all_preds.append(y_np)

    return np.concatenate(all_preds, axis=0)

# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Guitar Tabs from Audio")
    parser.add_argument("audio_file", type=str, help="Path to the .wav or .mp3 file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = paths.MODEL_DIR / "guitar_model.pth"

    if not model_path.exists():
        print(f"❌ Cannot find trained model at {model_path}. Did you run 2_train.py?")
        exit()

    model = GuitarTranscriberCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    features = extract_features(args.audio_file)
    if features is not None:
        raw_preds = predict(model, features, device)
        
        if raw_preds is not None:
            print("🪄 Applying Viterbi Physics Constraints...")
            decoder = GuitarViterbi(threshold=0.85)
            
            final_binary = np.zeros_like(raw_preds)
            for s in range(6):
                path = decoder.decode_string(raw_preds[:, s, :])
                for t, state in enumerate(path):
                    if state < 21: 
                        final_binary[t, s, state] = 1.0
            
            filename = os.path.basename(args.audio_file)
            tab_text = matrix_to_tab(final_binary, f"TRANSCRIBED: {filename}")
            
            out_file = paths.TAB_DIR / f"{filename}.txt"
            with open(out_file, "w") as f:
                f.write(tab_text)

            print("\n" + "="*50)
            print(f"✅ Transcription Complete!")
            print(f"📂 Saved to: {out_file}")
            print("="*50)