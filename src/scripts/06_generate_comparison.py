import torch
import numpy as np
import os
import scipy.ndimage as ndimage
from model import GuitarTranscriberCNN

# ==================================================
# CONFIG
# ==================================================
TEST_FILE = "./processed_data/test/05_Jazz3-137-Eb_comp.npz"
MODEL_PATH = "./saved_models/guitar_model.pth"

OUTPUT_DIR = r"E:/Old_CQT_Guitar_TAB/scripts/comparison_tab"

THRESHOLD  = 0.4
DOWNSAMPLE = 8

STRING_NAMES = ["e", "B", "G", "D", "A", "E"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================================================
# TAB CONVERSION
# ==================================================
def matrix_to_tab(binary_matrix, title, strings=None):
    """
    binary_matrix: (T, 6, 21)
    strings: list of string indices to render (default: all)
    """
    if strings is None:
        strings = list(range(6))

    tab_lines = {s: [f"{STRING_NAMES[s]}|"] for s in strings}
    T = binary_matrix.shape[0]

    for t in range(0, T, DOWNSAMPLE):
        for s in strings:
            visual = 5 - s
            fret_vec = binary_matrix[t, s]
            fret = fret_vec.argmax()

            if fret_vec[fret] > 0.5 and fret > 0:
                tab_lines[s].append(str(fret).ljust(2, "-"))
            else:
                tab_lines[s].append("--")

    text = f"\n--- {title} ---\n"
    for s in sorted(strings, reverse=True):
        text += "".join(tab_lines[s]) + "\n"
    return text


# ==================================================
# MAIN
# ==================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(TEST_FILE)

    print(f"Loading: {TEST_FILE}")
    data = np.load(TEST_FILE)
    cqt = data["cqt"]
    labels = data["labels"]

    # ---------- LOAD MODEL ----------
    model = GuitarTranscriberCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # ---------- INFERENCE ----------
    x = torch.from_numpy(cqt.T).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).squeeze().cpu().numpy()

    # ---------- POST PROCESS ----------
    pred = (probs > THRESHOLD).astype(np.float32)
    pred = ndimage.binary_closing(pred, structure=np.ones((4, 1, 1)))
    pred = ndimage.binary_opening(pred, structure=np.ones((2, 1, 1)))

    # ---------- GENERATE TABS ----------
    base = os.path.splitext(os.path.basename(TEST_FILE))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base}_comparison.txt")

    gt_tab   = matrix_to_tab(labels, "GROUND TRUTH (ACTUAL PLAYING)")
    pred_tab = matrix_to_tab(pred,   "MODEL PREDICTION")

    # Optional diagnostic: isolate string 6 (Low E)
    gt_s6   = matrix_to_tab(labels, "GT — STRING 6 (Low E)", strings=[5])
    pred_s6 = matrix_to_tab(pred,   "PRED — STRING 6 (Low E)", strings=[5])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(gt_tab)
        f.write("\n" + "=" * 60 + "\n")
        f.write(pred_tab)
        f.write("\n" + "=" * 60 + "\n")
        f.write(gt_s6)
        f.write("\n" + "=" * 60 + "\n")
        f.write(pred_s6)

    print(f"✅ Saved comparison TAB to:\n{out_path}")


# ==================================================
if __name__ == "__main__":
    main()
