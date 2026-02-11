import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from model import GuitarTranscriberCNN

# =========================
# CONFIG
# =========================
DATA_ROOT   = "./processed_data/test/"
MODEL_PATH  = "./saved_models/guitar_model.pth"
PLOT_DIR    = r"E:/Old_CQT_Guitar_TAB/scripts/plots"

THRESHOLD   = 0.7
CONTEXT     = 128
BATCH_SIZE  = 16

os.makedirs(PLOT_DIR, exist_ok=True)

# =========================
print("\n=== EVALUATION SCRIPT STARTED ===")

# =========================
# FIND TEST FILE
# =========================
files = sorted(glob.glob(os.path.join(DATA_ROOT, "*.npz")))
if not files:
    raise RuntimeError("❌ No test .npz files found.")

TEST_FILE = files[0]
basename  = os.path.basename(TEST_FILE).replace(".npz", "")
print(f"✅ Using test file: {TEST_FILE}")

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# =========================
# METRICS
# =========================
def calculate_metrics(y_true, y_pred):
    yt = y_true.flatten()
    yp = y_pred.flatten()
    return (
        precision_score(yt, yp, zero_division=0),
        recall_score(yt, yp, zero_division=0),
        f1_score(yt, yp, zero_division=0),
    )

def stringwise_f1(y_true, y_pred):
    f1s = []
    for s in range(6):
        yt = y_true[s*21:(s+1)*21].flatten()
        yp = y_pred[s*21:(s+1)*21].flatten()
        f1s.append(f1_score(yt, yp, zero_division=0))
    return np.array(f1s)

# =========================
# MAIN
# =========================
def main():
    # ---------- LOAD DATA ----------
    print("Loading data...")
    data   = np.load(TEST_FILE)
    cqt    = data["cqt"]
    labels = data["labels"]

    T, F = cqt.shape
    print(f"Data shape: {cqt.shape}")
    print(f"Total frames: {T}")

    # ---------- LOAD MODEL ----------
    model = GuitarTranscriberCNN().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=device, weights_only=True)
        )
        print("✅ Model loaded.")
    else:
        print("⚠️ Model not found.")

    model.eval()

    # =========================
    # INFERENCE
    # =========================
    n_chunks = T // CONTEXT
    print(f"Chunks: {n_chunks}, Batch size: {BATCH_SIZE}")

    all_preds = []

    with torch.no_grad():
        for i in tqdm(range(0, n_chunks, BATCH_SIZE), desc="Inference"):
            batch = [
                cqt[j*CONTEXT:(j+1)*CONTEXT].T
                for j in range(i, min(i+BATCH_SIZE, n_chunks))
            ]

            x = torch.from_numpy(np.stack(batch)).float().unsqueeze(1).to(device)
            probs = torch.sigmoid(model(x))
            all_preds.append(probs.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    preds = preds.reshape(-1, 126).T

    # =========================
    # POST-PROCESS
    # =========================
    binary_map = (preds > THRESHOLD).astype(np.float32)

    clean_map = ndimage.binary_closing(binary_map, structure=np.ones((1, 4)))
    clean_map = ndimage.binary_opening(clean_map, structure=np.ones((1, 2))).astype(float)

    true_map = labels[:clean_map.shape[1]].reshape(-1, 126).T

    # =========================
    # METRICS
    # =========================
    p, r, f1 = calculate_metrics(true_map, clean_map)
    f1_strings = stringwise_f1(true_map, clean_map)

    print("\n" + "="*45)
    print(f"RESULTS — {basename}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("String-wise F1:", np.round(f1_strings, 3))
    print("="*45)

    print(f"GT density:   {true_map.mean():.4f}")
    print(f"Pred density: {clean_map.mean():.4f}")

    # =========================
    # PLOT 1 — QUALITATIVE MAPS
    # =========================
    plt.figure(figsize=(15, 12))
    plt.subplot(3,1,1)
    plt.imshow(true_map, aspect="auto", origin="lower", cmap="magma")
    plt.title("Ground Truth")

    plt.subplot(3,1,2)
    plt.imshow(binary_map, aspect="auto", origin="lower", cmap="magma")
    plt.title(f"Raw Prediction (>{THRESHOLD})")

    plt.subplot(3,1,3)
    plt.imshow(clean_map, aspect="auto", origin="lower", cmap="magma")
    plt.title("Post-Processed Prediction")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{basename}_qualitative.png"), dpi=300)
    plt.show()

    # =========================
    # PLOT 2 — PR vs THRESHOLD
    # =========================
    thresholds = np.linspace(0.1, 0.9, 17)
    Ps, Rs, Fs = [], [], []

    for th in thresholds:
        bm = (preds > th).astype(float)
        cm = ndimage.binary_opening(
            ndimage.binary_closing(bm, np.ones((1,4))), np.ones((1,2))
        )
        p_, r_, f_ = calculate_metrics(true_map, cm)
        Ps.append(p_); Rs.append(r_); Fs.append(f_)

    plt.figure(figsize=(8,5))
    plt.plot(thresholds, Ps, label="Precision")
    plt.plot(thresholds, Rs, label="Recall")
    plt.plot(thresholds, Fs, label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision / Recall / F1 vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{basename}_threshold_curve.png"), dpi=300)
    plt.show()

    # =========================
    # PLOT 3 — STRING-WISE F1
    # =========================
    plt.figure(figsize=(7,4))
    plt.bar(range(1,7), f1_strings)
    plt.xlabel("String")
    plt.ylabel("F1-score")
    plt.title("String-wise F1 Score")
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{basename}_string_f1.png"), dpi=300)
    plt.show()

    print(f"📁 Plots saved to: {PLOT_DIR}")
    print("✅ Evaluation complete.")

# =========================
if __name__ == "__main__":
    main()
