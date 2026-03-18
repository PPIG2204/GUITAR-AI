# src/3_evaluate.py
import os
import time
import numpy as np
import torch
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

import config
import paths
from model import GuitarTranscriberCNN

# =========================
# EVALUATION SETTINGS
# =========================
THRESHOLD = 0.5  # Adjusted for higher POS_WEIGHT
N_STRINGS = 6
N_FRETS = 21

# =========================
# VITERBI PHYSICS ENGINE
# =========================
class GuitarViterbi:
    def __init__(self, n_frets=21, threshold=0.85):
        self.n_frets = n_frets
        self.n_states = n_frets + 1 
        self.silence_idx = n_frets
        self.threshold = threshold 
        
        # PHYSICS PENALTIES
        self.jump_penalty = 0.5    # Penalize large fret jumps
        self.onset_penalty = 0.2   # Discourage flickering
        self.sustain_bonus = 0.1   # Encourage holding notes
        
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

def apply_viterbi(prob):
    T = prob.shape[0]
    out = np.zeros_like(prob, dtype=np.float32)
    decoder = GuitarViterbi(threshold=THRESHOLD)
    for s in range(N_STRINGS):
        path = decoder.decode_string(prob[:, s, :])
        for t in range(T):
            state = path[t]
            if state < N_FRETS: 
                out[t, s, state] = 1.0
    return out

# =========================
# TAB GENERATOR
# =========================
def matrix_to_tab(binary_matrix, title):
    STRING_NAMES = ["e", "B", "G", "D", "A", "E"]
    tab_lines = {s: [f"{STRING_NAMES[s]}|"] for s in range(6)}
    
    # Downsample visually so it isn't 1000 characters wide
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
# MAIN EVALUATION
# =========================
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==================================================")
    print(f"📊 EVALUATING: {config.EXPERIMENT_NAME}")
    print(f"==================================================")

    # 1. Plot Training Loss
    loss_file = paths.PLOT_DIR / "train_loss.npy"
    if loss_file.exists():
        loss = np.load(loss_file)
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(loss) + 1), loss, marker='o', label="Training Loss")
        plt.title(f"Training Loss - {config.EXPERIMENT_NAME}")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.grid(True)
        plt.legend()
        plt.savefig(paths.PLOT_DIR / "loss_curve.png")
        plt.close()
        print("✅ Loss curve generated.")

    # 2. Load Test Data
    test_files = sorted(list(paths.TEST_DATA.glob("*.npz")))
    if not test_files:
        print("❌ No test files found in", paths.TEST_DATA)
        return

    # 3. Load Model
    model_path = paths.MODEL_DIR / "guitar_model.pth"
    model = GuitarTranscriberCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    Y_true, Y_raw, Y_vit = [], [], []
    
    # 4. Run Inference
    with torch.no_grad():
        for path in tqdm(test_files, desc="Testing Model"):
            data = np.load(path)
            # The QA Inspector's smart fallback
            if "features" in data:
                features = data["features"].copy()
            elif "cqt" in data:
                features = data["cqt"].copy()
            else:
                raise KeyError(f"Missing features in {path}")
            labels = data["labels"].reshape(-1, N_STRINGS, N_FRETS)
            
            T = features.shape[0]
            n_chunks = T // config.CONTEXT_LENGTH
            if n_chunks == 0: continue

            preds = []
            for i in range(0, n_chunks, config.BATCH_SIZE):
                batch = [features[j * config.CONTEXT_LENGTH:(j + 1) * config.CONTEXT_LENGTH].T 
                         for j in range(i, min(i + config.BATCH_SIZE, n_chunks))]
                x = torch.from_numpy(np.stack(batch)).float().unsqueeze(1).to(device)
                y = torch.sigmoid(model(x))
                preds.append(y.cpu().numpy())

            prob = np.concatenate(preds, axis=0).reshape(-1, N_STRINGS, N_FRETS)
            
            # Align sizes
            limit = min(len(prob), len(labels))
            prob = prob[:limit]
            labels = labels[:limit]

            # Decoding
            raw_binary = (prob > THRESHOLD).astype(np.float32)
            raw_binary = ndimage.binary_opening(ndimage.binary_closing(raw_binary, np.ones((4,1,1))), np.ones((2,1,1)))
            vit_binary = apply_viterbi(prob)

            Y_true.append(labels)
            Y_raw.append(raw_binary)
            Y_vit.append(vit_binary)

            # Generate ONE example TAB file for the first test file
            if path == test_files[0]:
                with open(paths.TAB_DIR / "example_comparison.txt", "w") as f:
                    f.write(matrix_to_tab(labels, "GROUND TRUTH (ACTUAL PLAYING)"))
                    f.write(matrix_to_tab(vit_binary, "MODEL PREDICTION (WITH VITERBI)"))

    # 5. Calculate Metrics
    Y_true = np.concatenate(Y_true, axis=0).flatten()
    Y_raw = np.concatenate(Y_raw, axis=0).flatten()
    Y_vit = np.concatenate(Y_vit, axis=0).flatten()

    print("\n==================================================")
    print("📈 FINAL EXPERIMENT RESULTS")
    print("==================================================")
    print(f"{'Method':<15} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 55)
    
    p_raw, r_raw, f1_raw = precision_score(Y_true, Y_raw, zero_division=0), recall_score(Y_true, Y_raw, zero_division=0), f1_score(Y_true, Y_raw, zero_division=0)
    print(f"{'Raw CNN':<15} | {p_raw:<10.4f} | {r_raw:<10.4f} | {f1_raw:<10.4f}")
    
    p_vit, r_vit, f1_vit = precision_score(Y_true, Y_vit, zero_division=0), recall_score(Y_true, Y_vit, zero_division=0), f1_score(Y_true, Y_vit, zero_division=0)
    print(f"{'Viterbi':<15} | {p_vit:<10.4f} | {r_vit:<10.4f} | {f1_vit:<10.4f}")
    print("==================================================")
    
    # Save CSV
    with open(paths.METRICS_DIR / "final_results.csv", "w") as f:
        f.write("Method,Precision,Recall,F1\n")
        f.write(f"Raw CNN,{p_raw:.4f},{r_raw:.4f},{f1_raw:.4f}\n")
        f.write(f"Viterbi,{p_vit:.4f},{r_vit:.4f},{f1_vit:.4f}\n")

if __name__ == "__main__":
    evaluate()
