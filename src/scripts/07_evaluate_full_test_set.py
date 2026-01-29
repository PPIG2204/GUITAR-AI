import os
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
CSV_PATH   = os.path.join(OUTPUT_DIR, "test_set_metrics.csv")

PLOT_DIR = r"E:\Old_CQT_Guitar_TAB\scripts\plots"

THRESHOLD  = 0.4
CONTEXT    = 128
BATCH_SIZE = 16

N_STRINGS = 6
N_FRETS   = 21

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
    return {
        "precision": precision_score(yt, yp, zero_division=0),
        "recall":    recall_score(yt, yp, zero_division=0),
        "f1":        f1_score(yt, yp, zero_division=0),
    }

def string_wise_f1(y_true, y_pred):
    return np.array([
        f1_score(
            y_true[:, s, :].flatten(),
            y_pred[:, s, :].flatten(),
            zero_division=0
        )
        for s in range(N_STRINGS)
    ])

def fret_wise_f1(y_true, y_pred):
    return np.array([
        f1_score(
            y_true[:, :, f].flatten(),
            y_pred[:, :, f].flatten(),
            zero_division=0
        )
        for f in range(N_FRETS)
    ])

# ==================================================
# INFERENCE
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

        y = torch.sigmoid(model(x))
        preds.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    preds = preds.reshape(-1, N_STRINGS, N_FRETS)
    return preds

# ==================================================
# POST PROCESS
# ==================================================
def post_process(preds):
    binary = (preds > THRESHOLD).astype(np.float32)
    binary = ndimage.binary_closing(binary, structure=np.ones((4,1,1)))
    binary = ndimage.binary_opening(binary, structure=np.ones((2,1,1)))
    return binary

# ==================================================
# PLOTS
# ==================================================
def plot_string_f1(f1_strings):
    plt.figure(figsize=(6,4))
    plt.bar(range(1, N_STRINGS+1), f1_strings)
    plt.xlabel("String index")
    plt.ylabel("F1-score")
    plt.title("String-wise F1-score")
    plt.ylim(0,1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "string_wise_f1.png"), dpi=300)
    plt.close()

def plot_fret_f1(f1_frets):
    plt.figure(figsize=(7,4))
    plt.plot(range(N_FRETS), f1_frets, marker="o")
    plt.xlabel("Fret index")
    plt.ylabel("F1-score")
    plt.title("Fret-wise F1-score")
    plt.ylim(0,1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "fret_wise_f1.png"), dpi=300)
    plt.close()

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

    print("="*60)
    print(f"Test files: {len(test_files)}")
    print(f"Device: {device}")
    print("="*60)

    # ---------- LOAD MODEL ----------
    model = GuitarTranscriberCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Model loaded\n")

    all_true, all_pred = [], []
    per_file_metrics = []

    # ---------- LOOP ----------
    for path in tqdm(test_files, desc="Evaluating", unit="file"):
        data = np.load(path)
        cqt    = data["cqt"]
        labels = data["labels"].reshape(-1, N_STRINGS, N_FRETS)

        preds = infer_file(model, cqt)
        if preds is None:
            continue

        preds = post_process(preds)

        T = min(len(preds), len(labels))
        preds, labels = preds[:T], labels[:T]

        all_true.append(labels)
        all_pred.append(preds)

        m = compute_metrics(labels, preds)
        m["file"] = os.path.basename(path)
        per_file_metrics.append(m)

    # ---------- AGGREGATE ----------
    Y_true = np.concatenate(all_true, axis=0)
    Y_pred = np.concatenate(all_pred, axis=0)

    overall    = compute_metrics(Y_true, Y_pred)
    f1_strings = string_wise_f1(Y_true, Y_pred)
    f1_frets   = fret_wise_f1(Y_true, Y_pred)

    # ---------- PRINT ----------
    print("\n" + "="*60)
    print("OVERALL TEST SET RESULTS")
    print(f"Precision: {overall['precision']:.4f}")
    print(f"Recall:    {overall['recall']:.4f}")
    print(f"F1-score:  {overall['f1']:.4f}")
    print("="*60)

    # ---------- SAVE CSV ----------
    with open(CSV_PATH, "w", encoding="utf-8") as f:
        f.write("file,precision,recall,f1\n")
        for m in per_file_metrics:
            f.write(f"{m['file']},{m['precision']:.4f},{m['recall']:.4f},{m['f1']:.4f}\n")
        f.write("\nOVERALL,,,\n")
        f.write(f"ALL,{overall['precision']:.4f},{overall['recall']:.4f},{overall['f1']:.4f}\n")

    # ---------- PLOTS ----------
    plot_string_f1(f1_strings)
    plot_fret_f1(f1_frets)

    print(f"📁 Metrics saved to: {CSV_PATH}")
    print(f"📊 Plots saved to:   {PLOT_DIR}")
    print("✅ Full test set evaluation complete.")

# ==================================================
if __name__ == "__main__":
    main()
