from pathlib import Path

# This finds the absolute path to the GUITAR-AI folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = BASE_DIR / "data"
RAW_AUDIO  = DATA_DIR / "audio_mono-mic"
HEX_AUDIO  = DATA_DIR / "audio_hex-pickup_debleeded"  # Your new source
ANNOTATION = DATA_DIR / "annotation"
ATOM_DIR   = DATA_DIR / "atoms"
SYNTH_DIR  = DATA_DIR / "synthetic_chords"

# Output Paths
RESULTS_DIR = BASE_DIR / "results"
PLOT_DIR    = RESULTS_DIR / "plots"
MODEL_DIR   = RESULTS_DIR / "saved_models"

# Ensure directories exist
for folder in [PLOT_DIR, MODEL_DIR, ATOM_DIR, SYNTH_DIR]:
    folder.mkdir(parents=True, exist_ok=True)