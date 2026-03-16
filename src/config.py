# src/config.py

# ==========================================
# EXPERIMENT CONTROL PANEL
# ==========================================
# Change this name for every new experiment! 
# (e.g., "Exp_Mel_NoSynth", "Exp_STFT_Synth")
EXPERIMENT_NAME = "Exp_STFT_NoSynth"  

# --- DATA SETTINGS ---
USE_SYNTHETIC_DATA = False

# --- AUDIO FEATURE EXTRACTION ---
# Options: 'CQT', 'MEL', 'STFT'
FEATURE_TYPE = 'STFT'  

SAMPLE_RATE = 22050
HOP_LENGTH = 512
CONTEXT_LENGTH = 128
STRIDE = 64

# CQT Specific
CQT_BINS = 192
BINS_PER_OCTAVE = 24

# Mel Specific
N_MELS = 128

# STFT Specific
N_FFT = 2048

# --- MODEL / TRAINING ---
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 1e-4
POS_WEIGHT = 15.0

# --- DETERMINISM ---
SEED = 42