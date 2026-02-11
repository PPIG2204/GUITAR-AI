import os
import numpy as np
import torch
import scipy.ndimage as ndimage
from tqdm import tqdm
from model import GuitarTranscriberCNN

# ==================================================
# CONFIG
# ==================================================
TEST_DIR   = "./processed_data/test"
MODEL_PATH = "./saved_models/guitar_model.pth"

OUTPUT_DIR = "./output_tab"

THRESHOLD   = 0.4
CONTEXT     = 128
BATCH_SIZE  = 16          # GTX 1660 sweet spot
DOWNSAMPLE  = 8

STRING_NAMES = ["e", "B", "G", "D", "A", "E"]

# ==================================================
# DEVICE
# ==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

print("=" * 60)
print(f"Running on device: {device}")
print("=" * 60)

# ==================================================
# TAB CONVERSION
# ==================================================
def matrix_to_tab(binary, downsample_rate):
    """
    binary: (T, 6, 21)
    """
    tab = {i: [f"{name}|"] for i, name in enumerate(STRING_NAMES)}
    T = binary.shape[0]

    for t in range(0, T, downsample_rate):
        frame = binary[t]

        for s in range(6):
            visual = 5 - s
            fret = frame[s].argmax()

            if frame[s, fret] and fret > 0:
                tab[visual].append(str(fret).ljust(2, "-"))
            else:
                tab[visual].append("--")

    return "\n".join("".join(tab[i]) for i in range(6))


# ==================================================
# INFERENCE
# ==================================================
@torch.no_grad()
def run_inference(model, cqt):
    """
    cqt: (T, F)
    return: (T, 6, 21)
    """
    T = cqt.shape[0]

    if T < CONTEXT:
        return np.zeros((0, 6, 21), dtype=np.float32)

    n_chunks = T // CONTEXT
    preds = []

    for i in tqdm(
        range(0, n_chunks, BATCH_SIZE),
        desc="  Inference",
        unit="batch",
        leave=False,
        dynamic_ncols=True
    ):
        batch = [
            cqt[j * CONTEXT:(j + 1) * CONTEXT].T
            for j in range(i, min(i + BATCH_SIZE, n_chunks))
        ]

        x = torch.from_numpy(np.stack(batch)) \
                .float() \
                .unsqueeze(1) \
                .to(device)

        y = torch.sigmoid(model(x))
        preds.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    return preds.reshape(-1, 6, 21)


# ==================================================
# POST PROCESS
# ==================================================
def post_process(preds):
    binary = (preds > THRESHOLD).astype(np.float32)

    # sustain
    binary = ndimage.binary_closing(
        binary, structure=np.ones((4, 1, 1))
    )
    # noise removal
    binary = ndimage.binary_opening(
        binary, structure=np.ones((2, 1, 1))
    )

    return binary.astype(np.float32)


# ==================================================
# PROCESS ONE FILE
# ==================================================
def process_file(model, file_path):
    base = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base}.tab.txt")

    data = np.load(file_path)
    cqt = data["cqt"]

    preds = run_inference(model, cqt)
    if preds.shape[0] == 0:
        print(f"  ⚠️ Skipped (too short): {base}")
        return

    binary = post_process(preds)
    tab_text = matrix_to_tab(binary, DOWNSAMPLE)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tab_text)

    print(f"  ✅ Saved → {out_path}")


# ==================================================
# MAIN
# ==================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    test_files = sorted([
        os.path.join(TEST_DIR, f)
        for f in os.listdir(TEST_DIR)
        if f.endswith(".npz")
    ])

    print(f"Found {len(test_files)} test files")

    # ---------- LOAD MODEL ----------
    model = GuitarTranscriberCNN().to(device)
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("✅ Model loaded\n")

    # ---------- PROCESS ALL ----------
    for file_path in tqdm(
        test_files,
        desc="Processing files",
        unit="file",
        dynamic_ncols=True
    ):
        print(f"\n▶ {os.path.basename(file_path)}")
        process_file(model, file_path)

    print("\n" + "=" * 60)
    print("✅ ALL TABS GENERATED")
    print(f"Output folder: {OUTPUT_DIR}")
    print("=" * 60)


# ==================================================
if __name__ == "__main__":
    main()
