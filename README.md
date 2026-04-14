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

## � Quick Start - Web Demo

If you just want to try the guitar transcription demo without training:

1. **Clone and setup:**
   ```bash
   git clone https://github.com/PPIG2204/GUITAR-AI.git
   cd GUITAR-AI
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   pip install -r requirements.txt
   ```

2. **Run the web demo:**
   ```bash
   python web_demo.py
   ```

3. **Open browser:** Go to `http://127.0.0.1:8501` and upload a guitar audio file!

> **Note:** The trained model checkpoint is included in the repository, so no training required!

---

## 📂 Project Structure

```bash
GUITAR-AI/
├── .vscode/                               # VS Code workspace settings
├── archieve/                              # Legacy scripts and utilities
│   └── legacy_scripts/
├── results/                               # Experiment outputs and saved models
│   └── Exp_CQT_GRU_HighWeight/
│       └── saved_models/                   # Trained model checkpoints
├── src/                                   # Source code and core project scripts
├── .gitignore
├── LICENSE
├── README.md
├── config.yaml
├── requirements.txt
└── web_demo.py                            # Local web server for tab generation
```

The `src/` folder remains unchanged from the existing project layout.

## File Descriptions

| File | Description |
| :--- | :--- |
| `src/1_preprocessing.py` | Preprocesses raw audio and JAMS files into training chunks. |
| `src/2_train.py` | Trains the GuitarTranscriberCNN model and saves the checkpoint. |
| `src/3_evaluate.py` | Evaluates model performance on the test split. |
| `src/4_inference.py` | Extracts audio features and runs inference with the trained model. |
| `src/config.py` | Defines configuration values for audio, model, and training. |
| `src/model.py` | Contains the neural network architecture. |
| `src/paths.py` | Defines project path locations for data and outputs. |
| `web_demo.py` | Starts a local web server for audio upload and tab generation. |

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

### Quick Web Demo (No Training Required)
If you just want to test the transcription with the pre-trained model:

```bash
# After installation above
python web_demo.py
# Then open http://127.0.0.1:8501 in your browser
```

### Full Pipeline (Training from Scratch)
### Full Pipeline (Training from Scratch)
If you want to retrain the model or run the complete pipeline:

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
python3 05_generate_tab.py 
```
### 4. Evaluation
```bash
python3 08_threshold_sweep.py
python3 09_constraint_decoding.py
```

> **Note:** After training, you can also run `python web_demo.py` from the project root to use the newly trained model.

