# GUITAR-AI: AI-Powered Guitar Tablature Generation

**GUITAR-AI** is a deep learning system designed to automatically transcribe guitar audio into digital tablature. By leveraging a Convolutional Neural Network (CNN) and advanced signal processing, it bridges the gap between raw audio recordings and readable musical notation.

## 🎸 Project Overview

Transcribing polyphonic guitar music is a complex task due to the ambiguity of same-pitch notes existing on different strings. This project solves that by:
1.  **Hearing:** Using Constant-Q Transform (CQT) to extract spectral features aligned with musical frequencies.
2.  **Understanding:** A CNN model predicts the probability of active notes across all 6 strings and 19 frets.
3.  **Refining:** Applying Viterbi decoding to enforce physical playability constraints (e.g., hand span, finger allocation).

## 🚀 Features

* **Polyphonic Pitch Estimation:** Detects multiple notes playing simultaneously.
* **String & Fret Identification:** Distinguishes *where* a note is played on the fretboard, not just the pitch.
* **Data Pipeline:** Automated scripts for processing the **GuitarSet** dataset.
* **Playability Constraints:** Post-processing algorithms to ensure generated tabs are human-playable.
* **Cross-Platform:** Developed and tested on Linux Mint and Windows.

## 🛠️ Tech Stack

* **Core:** Python 3.x
* **Deep Learning:** PyTorch
* **Audio Processing:** Librosa, NumPy
* **Data Handling:** JAMS (JSON Annotated Music Specification)
* **Tablature Output:** PyGuitarPro

## 📂 Project Structure

The project utilizes a numbered script pipeline located in `src/scripts/` to ensure a reproducible workflow:

| Script | Description |
| :--- | :--- |
| `00_check_environment.py` | Verifies GPU availability and required library versions. |
| `01_organize_files.py` | Splits **GuitarSet** into Train/Test sets (Player 05 reserved for testing). |
| `02_preprocessing.py` | Converts audio to CQT spectrograms and JAMS to label matrices (`.npz`). |
| `03_train.py` | Trains the CNN model and saves checkpoints. |
| `03b_plot_model.py` | Visualizes training loss and model architecture. |
| `04_plot_cqt.py` | Helper tool to inspect CQT features. |
| `05_generate_tab.py` | **Inference:** Generates tablature from new audio files. |
| `07_evaluate.py` | Calculates Precision, Recall, and F1 metrics on the test set. |
| `08_threshold_sweep.py` | Optimizes the probability threshold for note detection. |
| `09_constraint_decoding.py` | Compares `raw` model output vs. `viterbi` decoding. |

## ⚡ Usage

### 1. Setup & Data
Ensure you have the **GuitarSet** dataset downloaded. Run the environment check to confirm dependencies:
```bash
python src/scripts/00_check_environment.py
