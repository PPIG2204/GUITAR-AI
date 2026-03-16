from pathlib import Path

# This finds the GUITAR-AI root folder correctly
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Folders
DATA_DIR = BASE_DIR / "data"
RAW_AUDIO = DATA_DIR / "audio_mono-mic"
HEX_AUDIO = DATA_DIR / "audio_hex-pickup_debleeded"
ANNOTATION = DATA_DIR / "annotation"
SYNTH_DIR = DATA_DIR / "synthetic_chords"

# Preprocessed Output Folders
PROCESSED_DIR = BASE_DIR / "processed_data"
TRAIN_DATA = PROCESSED_DIR / "train"
SYNTH_DATA = PROCESSED_DIR / "synthetic"

# Results
RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR = RESULTS_DIR / "saved_models"
PLOT_DIR = RESULTS_DIR / "plots"

# Ensure all folders exist
for folder in [PROCESSED_DIR, TRAIN_DATA, SYNTH_DATA, MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)