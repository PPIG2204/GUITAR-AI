import os
import time
import numpy as np
import torch
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import sys

from tqdm import tqdm
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc
)

# 1. PATH INJECTION
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paths import MODEL_DIR, BASE_DIR, RESULTS_DIR, PLOT_DIR

from model import GuitarTranscriberCNN

# ==================================================
# CONFIG
# ==================================================
# Dynamically point to project root
TEST_DIR   = str(BASE_DIR / "processed_data" / "test") 
MODEL_PATH = str(MODEL_DIR / "guitar_model.pth")

OUTPUT_DIR = str(RESULTS_DIR)
# Fixed hardcoded E: drive path
TARGET_PLOT_DIR = str(PLOT_DIR) 

CSV_PATH = os.path.join(OUTPUT_DIR, "threshold_sweep_metrics.csv")

CONTEXT     = 128
BATCH_SIZE  = 16
N_STRINGS = 6
N_FRETS   = 21

# Aggressive sweep to catch the high-recall calibration point
THRESHOLDS = np.linspace(0.1, 0.9, 17)

USE_MORPHOLOGY = True 
USE_AMP        = True 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def infer_file(model, cqt):
    T = cqt.shape[0]
    n_chunks = T // CONTEXT
    if n_chunks == 0: return None, 0

    preds = []
    n_frames = 0
    for i in range(0, n_chunks, BATCH_SIZE):
        batch = [
            cqt[j * CONTEXT:(j + 1) * CONTEXT].T
            for j in range(i, min(i + BATCH_SIZE, n_chunks))
        ]
        x = torch.from_numpy(np.stack(batch)).float().unsqueeze(1).to(device)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            y = torch.sigmoid(model(x))
        preds.append(y.cpu().numpy())
        n_frames += y.shape[0] * CONTEXT

    preds = np.concatenate(preds, axis=0).reshape(-1, N_STRINGS, N_FRETS)
    return preds, n_frames

def post_process(binary):
    # Temporal smoothing to prevent "flickering" notes
    binary = ndimage.binary_closing(binary, structure=np.ones((4,1,1)))
    binary = ndimage.binary_opening(binary, structure=np.ones((2,1,1)))
    return binary

def compute_metrics(y_true, y_pred):
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    return (
        precision_score(yt, yp, zero_division=0),
        recall_score(yt, yp, zero_division=0),
        f1_score(yt, yp, zero_division=0)
    )

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TARGET_PLOT_DIR, exist_ok=True)

    test_files = sorted([os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.endswith(".npz")])
    if not test_files:
        print(f"❌ No test files found in {TEST_DIR}")
        return

    model = GuitarTranscriberCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_probs, all_labels = [], []
    t_start = time.time()

    for path in tqdm(test_files, desc="Running Inference"):
        data = np.load(path)
        probs, _ = infer_file(model, data["cqt"])
        if probs is None: continue
        
        T = min(len(probs), len(data["labels"]))
        all_probs.append(probs[:T])
        all_labels.append(data["labels"][:T].reshape(-1, N_STRINGS, N_FRETS))

    Y_prob = np.concatenate(all_probs, axis=0)
    Y_true = np.concatenate(all_labels, axis=0)

    rows, f1s = [], []
    for th in THRESHOLDS:
        binary = (Y_prob > th)
        if USE_MORPHOLOGY: binary = post_process(binary)
        p, r, f = compute_metrics(Y_true, binary)
        rows.append((th, p, r, f))
        f1s.append(f)

    best_idx = np.argmax(f1s)
    best_th = THRESHOLDS[best_idx]

    # Save Results
    with open(CSV_PATH, "w") as f:
        f.write("threshold,precision,recall,f1\n")
        for th, p, r, f1 in rows: f.write(f"{th:.2f},{p:.4f},{r:.4f},{f1:.4f}\n")

    # Plotting Precision-Recall Tradeoff
    plt.figure(figsize=(8,5))
    plt.plot(THRESHOLDS, f1s, 'b-o', label='F1 Score')
    plt.axvline(best_th, color='r', linestyle='--', label=f'Best Thresh: {best_th:.2f}')
    plt.title("Threshold Calibration for High-Fidelity Transcription")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(os.path.join(TARGET_PLOT_DIR, "threshold_sweep.png"))

    print(f"\n✅ Sweep Complete. Best F1 {f1s[best_idx]:.4f} at threshold {best_th:.2f}")

if __name__ == "__main__":
    main()