import os
import time
import numpy as np
import torch
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from model import GuitarTranscriberCNN

# ==================================================
# CONFIG
# ==================================================
TEST_DIR   = "./processed_data/test"
MODEL_PATH = "./saved_models/guitar_model.pth"

OUTPUT_DIR = "./results"
PLOT_DIR   = r"E:\Old_CQT_Guitar_TAB\scripts\plots"

CSV_PATH = os.path.join(OUTPUT_DIR, "constraint_decoding_metrics.csv")

CONTEXT     = 128
BATCH_SIZE  = 16           # GTX 1660 sweet spot
THRESHOLD   = 0.70         # best from script 08

N_STRINGS = 6
N_FRETS   = 21

USE_AMP = True
USE_MORPH = True           # Applied to raw, Viterbi has its own smoothing

# ==================================================
# DEVICE
# ==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ==================================================
# METRICS
# ==================================================
def compute_metrics(y_true, y_pred):
    yt = y_true.flatten()
    yp = y_pred.flatten()
    return (
        precision_score(yt, yp, zero_division=0),
        recall_score(yt, yp, zero_division=0),
        f1_score(yt, yp, zero_division=0)
    )

# ==================================================
# INFERENCE (RAW PROB)
# ==================================================
@torch.no_grad()
def infer_file(model, cqt):
    T = cqt.shape[0]
    n_chunks = T // CONTEXT
    if n_chunks == 0:
        return None

    preds = []

    for i in range(0, n_chunks, BATCH_SIZE):
        batch = [
            cqt[j*CONTEXT:(j+1)*CONTEXT].T
            for j in range(i, min(i+BATCH_SIZE, n_chunks))
        ]

        x = torch.from_numpy(np.stack(batch)) \
                .float().unsqueeze(1).to(device)

        with torch.amp.autocast('cuda', enabled=USE_AMP):
            y = torch.sigmoid(model(x))

        preds.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    return preds.reshape(-1, N_STRINGS, N_FRETS)

# ==================================================
# VITERBI LOGIC
# ==================================================
# ==================================================
# VITERBI LOGIC (Calibrated for Threshold 0.70)
# ==================================================
class GuitarViterbi:
    def __init__(self, n_frets=21, threshold=0.70):
        self.n_frets = n_frets
        self.n_states = n_frets + 1 
        self.silence_idx = n_frets
        self.threshold = threshold 
        
        # RE-CALIBRATED HEURISTICS
        # Since we are comparing e.g., 0.8 vs 0.7, the log-diff is small (~0.13).
        # We must lower the penalties to match this smaller "energy" scale.
        self.jump_penalty = 0.5    # Was 10.0 (Too strict for new scale)
        self.onset_penalty = 0.2   # Was 5.0  (Was blocking valid notes)
        self.sustain_bonus = 0.1   # Was 2.0  (Gentle encouragement)
        
    def decode_string(self, prob_matrix_1d):
        """
        Run Viterbi on a single string (T, 21)
        """
        T, F = prob_matrix_1d.shape
        
        # 1. EMISSION MATRIX
        # FIX: We set the "Silence" score exactly to the Threshold.
        # This forces the decoder to prefer silence unless the note probability 
        # explicitly exceeds the user's threshold (0.7).
        silence_scores = np.full((T, 1), self.threshold)
        
        emissions = np.hstack([prob_matrix_1d, silence_scores]) # (T, 22)
        
        # Log domain for numerical stability
        eps = 1e-6 
        log_emissions = np.log(emissions + eps)
        
        # 2. Initialize DP Tables
        path_probs = np.zeros((T, self.n_states))
        path_pointers = np.zeros((T, self.n_states), dtype=int)
        
        path_probs[0] = log_emissions[0]
        
        # 3. Transition Matrix
        trans_mat = np.zeros((self.n_states, self.n_states))
        
        fret_indices = np.arange(self.n_frets)
        for f_prev in range(self.n_frets):
            dist = np.abs(fret_indices - f_prev)
            trans_mat[f_prev, :self.n_frets] = - (dist * self.jump_penalty)
            trans_mat[f_prev, f_prev] += self.sustain_bonus 

        # Silence Transitions
        trans_mat[self.silence_idx, self.silence_idx] = 0        
        trans_mat[:self.n_frets, self.silence_idx]    = 0        
        trans_mat[self.silence_idx, :self.n_frets]    = -self.onset_penalty 

        # 4. Forward Pass
        for t in range(1, T):
            scores = path_probs[t-1][:, None] + trans_mat
            best_prev_scores = np.max(scores, axis=0)
            best_prev_states = np.argmax(scores, axis=0)
            path_probs[t] = best_prev_scores + log_emissions[t]
            path_pointers[t] = best_prev_states
            
        # 5. Backward Pass
        best_path = np.zeros(T, dtype=int)
        best_path[-1] = np.argmax(path_probs[-1])
        
        for t in range(T-2, -1, -1):
            best_path[t] = path_pointers[t+1, best_path[t+1]]
            
        return best_path

# ==================================================
# DECODING WRAPPERS
# ==================================================
def raw_threshold_decode(prob):
    binary = (prob > THRESHOLD).astype(np.float32)
    if USE_MORPH:
        binary = ndimage.binary_closing(binary, structure=np.ones((4,1,1)))
        binary = ndimage.binary_opening(binary, structure=np.ones((2,1,1)))
    return binary

def constraint_decode(prob):
    """
    Applies Viterbi Decoding string-by-string.
    """
    T = prob.shape[0]
    out = np.zeros_like(prob, dtype=np.float32)
    
    decoder = GuitarViterbi(n_frets=N_FRETS)
    
    # Iterate over each string (Polyphonic = 6 Monophonic channels)
    for s in range(N_STRINGS):
        string_prob = prob[:, s, :] # (T, 21)
        path = decoder.decode_string(string_prob)
        
        # Convert state indices back to one-hot matrix
        for t in range(T):
            state = path[t]
            if state < N_FRETS: # If not silence
                out[t, s, state] = 1.0
                
    return out

# ==================================================
# MAIN
# ==================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    test_files = sorted([
        os.path.join(TEST_DIR, f)
        for f in os.listdir(TEST_DIR)
        if f.endswith(".npz")
    ])

    print("="*70)
    print("🎸 GUITAR TAB TRANSCRIPTION – VITERBI EVALUATION")
    print("="*70)
    print(f"Test files        : {len(test_files)}")
    print(f"Context length    : {CONTEXT}")
    print(f"Batch size        : {BATCH_SIZE}")
    print(f"Threshold (Raw)   : {THRESHOLD}")
    print(f"Device            : {device}")
    print(f"AMP enabled       : {USE_AMP}")
    print("="*70)

    # ---------- LOAD MODEL ----------
    model = GuitarTranscriberCNN().to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()
    print("✅ Model loaded\n")

    Y_true, Y_raw, Y_cons = [], [], []

    t0 = time.time()
    total_frames = 0

    for path in tqdm(test_files, desc="Inference", unit="file"):
        data = np.load(path)
        cqt    = data["cqt"]
        labels = data["labels"].reshape(-1, N_STRINGS, N_FRETS)

        prob = infer_file(model, cqt)
        if prob is None:
            continue

        T = min(len(prob), len(labels))
        prob   = prob[:T]
        labels = labels[:T]

        # 1. Raw Threshold (with Morphology)
        raw  = raw_threshold_decode(prob)
        
        # 2. Viterbi Decoding (Physics constrained)
        cons = constraint_decode(prob)

        Y_true.append(labels)
        Y_raw.append(raw)
        Y_cons.append(cons)

        total_frames += T

    infer_time = time.time() - t0

    Y_true = np.concatenate(Y_true, axis=0)
    Y_raw  = np.concatenate(Y_raw, axis=0)
    Y_cons = np.concatenate(Y_cons, axis=0)

    # ==================================================
    # METRICS
    # ==================================================
    pr_raw, rc_raw, f1_raw = compute_metrics(Y_true, Y_raw)
    pr_con, rc_con, f1_con = compute_metrics(Y_true, Y_cons)

    # ==================================================
    # PRINT SUMMARY
    # ==================================================
    print("\n📈 Inference summary")
    print("-"*50)
    print(f"Total frames      : {total_frames:,}")
    print(f"Inference time    : {infer_time:.2f} sec")
    print(f"Frames / second   : {total_frames / infer_time:,.1f}")
    print("-"*50)

    print("\n======================================================================")
    print("📊 COMPARISON: RAW (MORPH) vs VITERBI DECODING")
    print("======================================================================")
    print(f"{'Method':<20}{'Precision':>10}{'Recall':>10}{'F1':>10}")
    print("-"*50)
    print(f"{'Raw (+Morph)':<20}{pr_raw:10.4f}{rc_raw:10.4f}{f1_raw:10.4f}")
    print(f"{'Viterbi':<20}{pr_con:10.4f}{rc_con:10.4f}{f1_con:10.4f}")
    print("="*70)

    # ==================================================
    # SAVE CSV
    # ==================================================
    with open(CSV_PATH, "w", encoding="utf-8") as f:
        f.write("method,precision,recall,f1\n")
        f.write(f"raw,{pr_raw:.4f},{rc_raw:.4f},{f1_raw:.4f}\n")
        f.write(f"viterbi,{pr_con:.4f},{rc_con:.4f},{f1_con:.4f}\n")

    # ==================================================
    # BAR PLOT
    # ==================================================
    plt.figure(figsize=(6,4))
    plt.bar(["Raw (+Morph)", "Viterbi"], [f1_raw, f1_con], color=['gray', 'green'])
    plt.ylabel("F1-score")
    plt.title("Effect of Physics-Constrained Viterbi")
    plt.ylim(0,1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "constraint_vs_raw_f1.png"), dpi=300)
    plt.close()

    print("📁 CSV saved to :", CSV_PATH)
    print("📊 Plot saved to:", PLOT_DIR)
    print("✅ Evaluation complete.")
    print("="*70)

# ==================================================
if __name__ == "__main__":
    main()