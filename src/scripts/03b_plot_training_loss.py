import os
import numpy as np
import matplotlib.pyplot as plt

# =========================
# PATHS (ABSOLUTE, SAFE)
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = SCRIPT_DIR  # scripts/
LOSS_PATH = os.path.join(ROOT_DIR, "saved_models", "train_loss.npy")

OUT_DIR = r"E:\Old_CQT_Guitar_TAB\scripts\plots"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# LOAD + CHECK
# =========================
if not os.path.exists(LOSS_PATH):
    raise FileNotFoundError(
        f"❌ Missing: {LOSS_PATH}\n"
        "👉 You must save train_loss.npy during training."
    )

loss = np.load(LOSS_PATH)

assert loss.ndim == 1, "Loss must be 1D (epoch-wise)"
assert len(loss) > 1, "Need at least 2 epochs"

epochs = np.arange(1, len(loss) + 1)

# =========================
# EMA SMOOTHING
# =========================
def ema(x, alpha=0.3):
    y = [x[0]]
    for v in x[1:]:
        y.append(alpha * v + (1 - alpha) * y[-1])
    return np.array(y)

# =========================
# PLOT
# =========================
plt.figure(figsize=(8, 5))
plt.plot(epochs, loss, marker="o", label="Raw")
plt.plot(epochs, ema(loss), linestyle="--", label="EMA")

plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.suptitle("CNN Guitar TAB Transcription", fontsize=11)
plt.title("Training Loss (pos_weight=15, context=128)", fontsize=9)
plt.grid(True)
plt.legend()

# =========================
# SAVE
# =========================
out_png = os.path.join(OUT_DIR, "training_loss.png")
out_csv = os.path.join(OUT_DIR, "training_loss.csv")

plt.tight_layout()
plt.savefig(out_png, dpi=200)
plt.close()

np.savetxt(
    out_csv,
    np.c_[epochs, loss],
    delimiter=",",
    header="epoch,loss",
    comments=""
)

print("✅ Plot saved:", out_png)
print("✅ CSV saved :", out_csv)
