import os
import shutil
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_AUDIO = BASE_DIR / "data" / "audio_mono-mic"
SOURCE_JAMS  = BASE_DIR / "data" / "annotation"


PLAYER_TEST_ID = "05"

DIRS = {
    "train": {"audio": "./data/train/audio/", "jams": "./data/train/jams/"},
    "test":  {"audio": "./data/test/audio/",  "jams": "./data/test/jams/"}
}

# =========================
# SETUP
# =========================
for split in DIRS.values():
    os.makedirs(split["audio"], exist_ok=True)
    os.makedirs(split["jams"], exist_ok=True)

print("\n=== DATA ORGANIZATION STARTED ===")
print(f"Test player ID: {PLAYER_TEST_ID}")

# =========================
# HELPERS
# =========================
def split_mode(filename: str) -> str:
    """Return 'train' or 'test' based on player ID in filename"""
    return "test" if filename[:2] == PLAYER_TEST_ID else "train"

def copy_files(src_dir, dst_key, ext, counter):
    files = [f for f in os.listdir(src_dir) if f.endswith(ext)]
    
    if not files:
        print(f"[WARNING] No '{ext}' files found in {src_dir}")
        return

    desc = f"Copying {dst_key.upper():5s}"
    for f in tqdm(files, desc=desc, unit="file"):
        mode = split_mode(f)
        shutil.copy(
            os.path.join(src_dir, f),
            os.path.join(DIRS[mode][dst_key], f)
        )
        counter[dst_key][mode] += 1

# =========================
# MAIN LOGIC
# =========================
counter = {
    "audio": defaultdict(int),
    "jams":  defaultdict(int)
}

copy_files(SOURCE_AUDIO, "audio", ".wav", counter)
copy_files(SOURCE_JAMS,  "jams",  ".jams", counter)

# =========================
# SUMMARY & SANITY CHECKS
# =========================
print("\n=== DATA SPLIT SUMMARY ===")
print(f"AUDIO | Train: {counter['audio']['train']:3d} | "
      f"Test ({PLAYER_TEST_ID}): {counter['audio']['test']:3d}")
print(f"JAMS  | Train: {counter['jams']['train']:3d} | "
      f"Test ({PLAYER_TEST_ID}): {counter['jams']['test']:3d}")
print("==========================")

# Hard safety checks (research-grade)
assert counter["audio"]["test"] > 0, "❌ Test AUDIO set is empty!"
assert counter["jams"]["test"]  > 0, "❌ Test JAMS set is empty!"
assert counter["audio"]["train"] >= counter["audio"]["test"], \
       "❌ Train set smaller than test set!"

print("\n✅ Organization Complete. No leakage detected.\n")
