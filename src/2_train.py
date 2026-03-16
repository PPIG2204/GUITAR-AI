# src/2_train.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import config
import paths
from model import GuitarTranscriberCNN

# =========================
# REPRODUCIBILITY
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

set_seed(config.SEED)

# =========================
# DYNAMIC DATASET
# =========================
class GuitarDataset(Dataset):
    def __init__(self, data_dirs, context_width):
        self.context = context_width
        self.files = []
        
        for d_dir in data_dirs:
            if not d_dir.exists():
                continue
            print(f"📂 Indexing files in {d_dir.name}...")
            found = sorted([d_dir / f for f in os.listdir(d_dir) if f.endswith(".npz")])
            self.files.extend(found)

        print(f"🧩 Total mixed training chunks ready: {len(self.files):,}")

    def __getitem__(self, idx):
        path = self.files[idx]
        with np.load(path) as d:
            if "features" in d:
                features = d["features"].copy()
            elif "cqt" in d:
                features = d["cqt"].copy()
            else:
                raise KeyError(f"Missing features in {path}")
                
            labels = d["labels"].copy()

        # =======================================
        # THE SMART RUTHLESS ENFORCER
        # =======================================
        # 1. If it's too long, chop it off (Slice)
        features = features[:self.context]
        labels = labels[:self.context]

        # 2. Measure and pad FEATURES
        feat_len = features.shape[0]
        if feat_len < self.context:
            pad_amount = self.context - feat_len
            features = np.pad(features, ((0, pad_amount), (0, 0)), mode='constant')

        # 3. Measure and pad LABELS independently
        label_len = labels.shape[0]
        if label_len < self.context:
            pad_amount = self.context - label_len
            labels = np.pad(labels, ((0, pad_amount), (0, 0), (0, 0)), mode='constant')

        # CNN expects (Channels, Freq, Time)
        # Force the slices into clean, contiguous memory blocks
        features = torch.from_numpy(np.ascontiguousarray(features.T)).float().unsqueeze(0)
        labels = torch.from_numpy(np.ascontiguousarray(labels)).float()
        
        return features, labels

    def __len__(self):
        return len(self.files)

# =========================
# MAIN TRAINING LOOP
# =========================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==================================================")
    print(f"🚀 TRAINING EXPERIMENT: {config.EXPERIMENT_NAME}")
    print(f"⚙️  Feature: {config.FEATURE_TYPE} | Synth: {config.USE_SYNTHETIC_DATA} | Device: {device}")
    print(f"==================================================")

    # Decide which folders to pull data from based on config
    active_data_dirs = [paths.TRAIN_DATA]
    if config.USE_SYNTHETIC_DATA:
        active_data_dirs.append(paths.SYNTH_DATA)

    dataset = GuitarDataset(active_data_dirs, config.CONTEXT_LENGTH)
    
    if len(dataset) == 0:
        print("❌ Error: No training data found. Did you run 1_preprocess.py?")
        return

    loader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
        num_workers=4, pin_memory=True, drop_last=True
    )

    model = GuitarTranscriberCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.POS_WEIGHT, device=device))

    epoch_losses = []

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{config.EPOCHS}", unit="batch", dynamic_ncols=True)

        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(loader)
        epoch_losses.append(avg_loss)

    # =========================
    # ISOLATED SAVING
    # =========================
    model_path = paths.MODEL_DIR / "guitar_model.pth"
    loss_path  = paths.PLOT_DIR / "train_loss.npy"

    torch.save(model.state_dict(), model_path)
    np.save(loss_path, np.array(epoch_losses, dtype=np.float32))

    print(f"\n✅ Training Complete. Model securely locked in the vault.")
    print(f"💾 {model_path}")

if __name__ == "__main__":
    train()