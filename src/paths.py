# src/paths.py
import os
from pathlib import Path
from config import EXPERIMENT_NAME

# Base Project Root
BASE_DIR = Path(__file__).resolve().parent.parent

# Input Data
DATA_DIR = BASE_DIR / "data"
RAW_AUDIO = DATA_DIR / "audio_mono-mic"
ANNOTATION = DATA_DIR / "annotation"
SYNTH_DIR = DATA_DIR / "synthetic_chords"

# Experiment Outputs (Isolated by experiment name)
EXP_DIR = BASE_DIR / "results" / EXPERIMENT_NAME
MODEL_DIR = EXP_DIR / "saved_models"
PLOT_DIR = EXP_DIR / "plots"
METRICS_DIR = EXP_DIR / "metrics"
TAB_DIR = EXP_DIR / "generated_tabs"

# Processed Data (Now correctly nested inside EXP_DIR)
PROCESSED_DIR = EXP_DIR / "processed_data"
TRAIN_DATA = PROCESSED_DIR / "train"
TEST_DATA = PROCESSED_DIR / "test"
SYNTH_DATA = PROCESSED_DIR / "synthetic"

# Auto-create necessary output directories for the active experiment
for folder in [PROCESSED_DIR, TRAIN_DATA, TEST_DATA, SYNTH_DATA, MODEL_DIR, PLOT_DIR, METRICS_DIR, TAB_DIR]:
    folder.mkdir(parents=True, exist_ok=True)