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
USE_MORPH = True

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

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            y = torch.sigmoid(model(x))

        preds.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    return preds.reshape(-1, N_STRINGS, N_FRETS)

# ==================================================
# DECODING
# ==================================================
def raw_threshold_decode(prob):
    binary = (prob > THRESHOLD).astype(np.float32)
    if USE_MORPH:
        binary = ndimage.binary_closing(binary, structure=np.ones((4,1,1)))
        binary = ndimage.binary_opening(binary, structure=np.ones((2,1,1)))
    return binary

def constraint_decode(prob):
    """
    Enforce: at most 1 fret per string per frame
    """
    T = prob.shape[0]
    out = np.zeros_like(prob, dtype=np.float32)

    # argmax over fret axis
    max_fret = prob.argmax(axis=2)          # (T, 6)
    max_prob = prob.max(axis=2)              # (T, 6)

    mask = max_prob > THRESHOLD

    for s in range(N_STRINGS):
        idx = np.where(mask[:, s])[0]
        out[idx, s, max_fret[idx, s]] = 1.0

    if USE_MORPH:
        out = ndimage.binary_closing(out, structure=np.ones((4,1,1)))
        out = ndimage.binary_opening(out, structure=np.ones((2,1,1)))

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
    print("🎸 GUITAR TAB TRANSCRIPTION – CONSTRAINT DECODING EVAL")
    print("="*70)
    print(f"Test files        : {len(test_files)}")
    print(f"Context length    : {CONTEXT}")
    print(f"Batch size        : {BATCH_SIZE}")
    print(f"Threshold         : {THRESHOLD}")
    print(f"Device            : {device}")
    print(f"AMP enabled       : {USE_AMP}")
    print(f"Morphology        : {USE_MORPH}")
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

        raw  = raw_threshold_decode(prob)
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
    print("📊 COMPARISON: RAW vs CONSTRAINT DECODING")
    print("======================================================================")
    print(f"{'Method':<20}{'Precision':>10}{'Recall':>10}{'F1':>10}")
    print("-"*50)
    print(f"{'Raw threshold':<20}{pr_raw:10.4f}{rc_raw:10.4f}{f1_raw:10.4f}")
    print(f"{'Constraint':<20}{pr_con:10.4f}{rc_con:10.4f}{f1_con:10.4f}")
    print("="*70)

    # ==================================================
    # SAVE CSV
    # ==================================================
    with open(CSV_PATH, "w", encoding="utf-8") as f:
        f.write("method,precision,recall,f1\n")
        f.write(f"raw,{pr_raw:.4f},{rc_raw:.4f},{f1_raw:.4f}\n")
        f.write(f"constraint,{pr_con:.4f},{rc_con:.4f},{f1_con:.4f}\n")

    # ==================================================
    # BAR PLOT
    # ==================================================
    plt.figure(figsize=(6,4))
    plt.bar(["Raw", "Constraint"], [f1_raw, f1_con])
    plt.ylabel("F1-score")
    plt.title("Effect of Constraint Decoding")
    plt.ylim(0,1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "constraint_vs_raw_f1.png"), dpi=300)
    plt.close()

    print("📁 CSV saved to :", CSV_PATH)
    print("📊 Plot saved to:", PLOT_DIR)
    print("✅ Constraint decoding evaluation complete.")
    print("="*70)

# ==================================================
if __name__ == "__main__":
    main()
