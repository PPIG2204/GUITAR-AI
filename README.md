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

```bash
GUITAR-AI/
├── data/               # Raw audio and annotations (JAMS files)
├── src/
│   ├── models/         # Neural network architectures
│   └── scripts/        # Data preprocessing and utility scripts
├── output_tab/         # Generated tablature results
├── config.yaml         # Project configurations
└── requirements.txt    # Python dependencies
```

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


# How to Run GUITAR-AI (Windows & Linux)

This guide provides step-by-step instructions for setting up and running the **GUITAR-AI** pipeline on both Windows and Linux environments.

## 📋 Prerequisites

Before starting, ensure you have the following installed:
* **Git:** [Download Here](https://git-scm.com/downloads)
* **Python 3.8+:** [Download Here](https://www.python.org/downloads/)
---

## 🛠️ 1. Installation & Setup

### Step 1: Clone the Repository
Open your terminal (Linux) or Command Prompt/PowerShell (Windows) and run:

```bash
git clone https://github.com/PPIG2204/GUITAR-AI.git
cd GUITAR-AI
```
### Step 2: Download Dataset
1. **Source**: [Guitarset on Zanodo](https://zenodo.org/records/3371780)

2. **Files needed:** annotation.zip and audio_mono-mic.zip

3. **Placement:** Create a folder at GUITAR-AI/data/ and extract both zip files there.

### Step 3: Create a Virtual Environment
#### Linux
```bash
python3 -m venv venv
source venv/bin/activate
```
#### Windows
```bash
python -m venv venv
.\venv\Scripts\activate
```
### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```
#### FFmpeg: Required for audio processing.

#### Linux:
```bash 
sudo apt install ffmpeg libsndfile1
```

#### Windows: 
Download via [Gyan.dev](https://www.gyan.dev/ffmpeg/) and add the bin folder to your System Path.

## 2. Running the scripts
### 0. Change Directory
``` bash
cd src/scripts
```
### 1. Data Organization & Preprocessing
Organize the raw GuitarSet files and generate spectral features:
``` bash
python3 01_organize_files.py
python3 02_preprocessing.py
```
### 2. Training
```bash
python3 03_train.py
```

### 3. Generate Tablature (Inference)
```bash
python3 05_generate_tab.py --file path/to/your_audio.wav
```
* **For example:**
```bash
python3 05_generate_tab.py --file ../../data/audio_mono-mic/Deptrai_solo.wav
```
* The result tablature will be in GUITAR-AI/output_tab/
### 4. Evaluation
```bash
python3 08_threshold_sweep.py
python3 09_constraint_decoding.py
```

