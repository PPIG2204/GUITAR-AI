import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import GuitarTranscriberCNN

# =========================
# CONFIG
# =========================
TRAIN_DIR = "./processed_data/train/"
OUT_DIR = "./saved_models"

BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
POS_WEIGHT = 15.0

CONTEXT = 128
NUM_WORKERS = 4   # sweet spot for i7-9700
SEED = 42

# =========================
# REPRODUCIBILITY
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

set_seed(SEED)

# =========================
# DATASET
# =========================
class GuitarDataset(Dataset):
    def __init__(self, data_dir, context_width=128):
        self.context = context_width
        self.index = []

        self.files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".npz")
        )

        print(f"📂 Indexing {len(self.files)} files...")

        for f in self.files:
            with np.load(f) as d:
                n_frames = d["cqt"].shape[0]

            n_chunks = n_frames // context_width
            for i in range(n_chunks):
                self.index.append((f, i * context_width))

        print(f"🧩 Total training chunks: {len(self.index):,}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        path, start = self.index[idx]

        with np.load(path) as d:
            cqt = d["cqt"][start:start+self.context]     # (T, F)
            lbl = d["labels"][start:start+self.context]  # (T, 6, 21)

        # (1, F, T)
        cqt = torch.from_numpy(cqt.T).float().unsqueeze(0)
        lbl = torch.from_numpy(lbl).float()

        return cqt, lbl

# =========================
# TRAIN LOOP
# =========================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Training on: {device}")

    dataset = GuitarDataset(TRAIN_DIR, CONTEXT)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    model = GuitarTranscriberCNN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(POS_WEIGHT, device=device)
    )

    epoch_losses = []

    print("\n================ TRAINING =================")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch}/{EPOCHS}",
            unit="batch",
            dynamic_ncols=True
        )

        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            out = model(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(loader)
        epoch_losses.append(avg_loss)

        print(f"✅ Epoch {epoch:02d} | Avg loss: {avg_loss:.4f}")

    # =========================
    # SAVE ARTIFACTS
    # =========================
    os.makedirs(OUT_DIR, exist_ok=True)

    model_path = os.path.join(OUT_DIR, "guitar_model.pth")
    loss_path  = os.path.join(OUT_DIR, "train_loss.npy")

    torch.save(model.state_dict(), model_path)
    np.save(loss_path, np.array(epoch_losses, dtype=np.float32))

    print("\n================ SAVED =================")
    print(f"💾 Model     : {model_path}")
    print(f"📉 Loss log  : {loss_path}")
    print("=======================================\n")

# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    train()
