import os
import time
import numpy as np
import torch
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc
)

from model import GuitarTranscriberCNN

# ==================================================
# CONFIG
# ==================================================
TEST_DIR   = "./processed_data/test"
MODEL_PATH = "./saved_models/guitar_model.pth"

OUTPUT_DIR = "./results"
PLOT_DIR   = r"E:\Old_CQT_Guitar_TAB\scripts\plots"

CSV_PATH = os.path.join(OUTPUT_DIR, "threshold_sweep_metrics.csv")

CONTEXT     = 128
BATCH_SIZE  = 16

N_STRINGS = 6
N_FRETS   = 21

THRESHOLDS = np.linspace(0.05, 0.95, 19)

USE_MORPHOLOGY = True   # set False if benchmarking speed only
USE_AMP        = True   # mixed precision inference

# ==================================================
# DEVICE
# ==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ==================================================
# INFERENCE (RAW PROBS)
# ==================================================
@torch.no_grad()
def infer_file(model, cqt):
    T = cqt.shape[0]
    n_chunks = T // CONTEXT
    if n_chunks == 0:
        return None, 0

    preds = []
    n_frames = 0

    for i in range(0, n_chunks, BATCH_SIZE):
        batch = [
            cqt[j * CONTEXT:(j + 1) * CONTEXT].T
            for j in range(i, min(i + BATCH_SIZE, n_chunks))
        ]

        x = torch.from_numpy(np.stack(batch)) \
                 .float().unsqueeze(1).to(device)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            y = torch.sigmoid(model(x))

        preds.append(y.cpu().numpy())
        n_frames += y.shape[0] * CONTEXT

    preds = np.concatenate(preds, axis=0)
    preds = preds.reshape(-1, N_STRINGS, N_FRETS)

    return preds, n_frames

# ==================================================
# POST PROCESS (OPTIONAL)
# ==================================================
def post_process(binary):
    binary = ndimage.binary_closing(binary, structure=np.ones((4,1,1)))
    binary = ndimage.binary_opening(binary, structure=np.ones((2,1,1)))
    return binary

# ==================================================
# METRICS
# ==================================================
def compute_metrics(y_true, y_pred):
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    return (
        precision_score(yt, yp, zero_division=0),
        recall_score(yt, yp, zero_division=0),
        f1_score(yt, yp, zero_division=0)
    )

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

    print("\n" + "="*70)
    print("🎸 GUITAR TAB TRANSCRIPTION – FULL TEST EVALUATION")
    print("="*70)
    print(f"Test files        : {len(test_files)}")
    print(f"Context length    : {CONTEXT}")
    print(f"Batch size        : {BATCH_SIZE}")
    print(f"Thresholds        : {len(THRESHOLDS)}")
    print(f"Device            : {device}")
    print(f"AMP enabled       : {USE_AMP}")
    print(f"Morphology        : {USE_MORPHOLOGY}")
    print("="*70)

    # ---------- LOAD MODEL ----------
    model = GuitarTranscriberCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Model loaded")

    # ---------- COLLECT RAW ----------
    all_probs  = []
    all_labels = []

    total_frames = 0
    t_start = time.time()

    for path in tqdm(test_files, desc="Inference", unit="file"):
        data = np.load(path)
        cqt    = data["cqt"]
        labels = data["labels"].reshape(-1, N_STRINGS, N_FRETS)

        probs, n_frames = infer_file(model, cqt)
        if probs is None:
            continue

        T = min(len(probs), len(labels))
        all_probs.append(probs[:T])
        all_labels.append(labels[:T])
        total_frames += n_frames

    infer_time = time.time() - t_start

    Y_prob = np.concatenate(all_probs, axis=0)
    Y_true = np.concatenate(all_labels, axis=0)

    print("\n📈 Inference summary")
    print("-" * 50)
    print(f"Total frames      : {total_frames:,}")
    print(f"Inference time    : {infer_time:.2f} sec")
    print(f"Frames / second   : {total_frames / infer_time:,.1f}")
    print(f"Y_prob shape      : {Y_prob.shape}")
    print("-" * 50)

    yt_flat = Y_true.reshape(-1)
    yp_flat = Y_prob.reshape(-1)

    # ==================================================
    # THRESHOLD SWEEP
    # ==================================================
    rows = []
    precisions, recalls, f1s = [], [], []

    for th in THRESHOLDS:
        binary = (Y_prob > th)

        if USE_MORPHOLOGY:
            binary = post_process(binary)

        p, r, f = compute_metrics(Y_true, binary)

        rows.append((th, p, r, f))
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    best_idx = int(np.argmax(f1s))
    best_th  = THRESHOLDS[best_idx]

    # ---------- SAVE CSV ----------
    with open(CSV_PATH, "w", encoding="utf-8") as f:
        f.write("threshold,precision,recall,f1\n")
        for th, p, r, f1 in rows:
            f.write(f"{th:.2f},{p:.4f},{r:.4f},{f1:.4f}\n")

    # ==================================================
    # PLOTS
    # ==================================================
    plt.figure(figsize=(6,4))
    plt.plot(THRESHOLDS, f1s, marker="o")
    plt.axvline(best_th, linestyle="--", alpha=0.6)
    plt.xlabel("Threshold")
    plt.ylabel("F1-score")
    plt.title("F1-score vs Decision Threshold")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "f1_vs_threshold.png"), dpi=300)
    plt.close()

    prec, rec, _ = precision_recall_curve(yt_flat, yp_flat)
    pr_auc = auc(rec, prec)

    plt.figure(figsize=(6,4))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve (AUC={pr_auc:.3f})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "precision_recall_curve.png"), dpi=300)
    plt.close()

    # ==================================================
    # FINAL CONSOLE SUMMARY
    # ==================================================
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)
    print(f"Best threshold    : {best_th:.2f}")
    print(f"Best Precision    : {precisions[best_idx]:.4f}")
    print(f"Best Recall       : {recalls[best_idx]:.4f}")
    print(f"Best F1-score     : {f1s[best_idx]:.4f}")
    print(f"PR-AUC            : {pr_auc:.4f}")
    print("-" * 70)
    print(f"CSV saved to      : {CSV_PATH}")
    print(f"Plots saved to    : {PLOT_DIR}")
    print("="*70)

# ==================================================
if __name__ == "__main__":
    main()
