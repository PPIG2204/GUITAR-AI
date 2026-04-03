# GUITAR-AI: Polyphonic CRNN Transcription Engine

**Project Status:** Archived (Production-Ready)  
**Architecture:** 3-Layer CNN + 2-Layer Bidirectional GRU + Viterbi Physics Engine  

## 📌 The "North Star" Objective
While the associated academic report (*"Ràng buộc chuyển tiếp khả vi..."*) restricted its scope to monophonic (single-note) prediction to simplify mathematical modeling, **GUITAR-AI** was engineered to solve the practical challenge of **full polyphony**. This system transcribes real-world guitar chords and complex melodies into 6-string tablature, enforcing physical playability through a post-hoc Viterbi-decoding "Physics Bouncer" rather than purely theoretical loss functions.

## 🛠️ Key Engineering Patches (Final Build)
Unlike the base models used in the academic study, this archived version includes critical engineering fixes:
* **The CQT Noise Floor:** Implemented a strict **80dB noise floor gate** in `1_preprocess.py` to eliminate "silence hallucinations" and ghost notes common in high-gain audio.
* **The Open-String Fix:** Patched the quantization logic in `3_evaluate.py` to properly retain **Fret 0** (open strings), preventing the model from erroneously deleting fundamental chord components.
* **The Physics Engine:** Integrated a Viterbi decoding algorithm tuned with a **0.5 sustain bonus** to eliminate note flickering and ensure anatomically possible finger transitions.

## 🏗️ Technical Architecture
Developed for high-performance execution on consumer-grade hardware (Local execution optimized).

1.  **Preprocessing (`1_preprocess.py`):** Extracts Constant-Q Transform (CQT) spectrograms (192 bins, 24 bins/octave) with log-scale normalization.
2.  **Neural Core (`model.py`):** A Convolutional Recurrent Neural Network (CRNN) that combines CNN spatial feature extraction with a **Bidirectional GRU** for temporal memory, predicting probabilities for 6 strings across 21 frets simultaneously.
3.  **Decoding & Eval (`3_evaluate.py`):** Converts raw probabilities into quantized tablature files using a physics-aware Viterbi engine.

## 📂 Project Structure
```bash
GUITAR-AI/
├── src/
│   ├── 1_preprocess.py  # Feature extraction & noise gating
│   ├── 2_train.py       # CRNN training with Adam optimizer
│   ├── 3_evaluate.py    # Viterbi decoding & metrics
│   ├── model.py         # Bi-GRU + CNN Architecture
│   ├── config.py        # Experiment control panel
│   └── paths.py         # Dynamic path management
└── results/             # Isolated experiment logs, plots, and metrics
```

## 🚀 Execution Pipeline
1.  **Configuration:** Define your experiment in `src/config.py` (e.g., `Exp_CQT_GRU_HighWeight`).
2.  **Process:**
    ```bash
    python src/1_preprocess.py
    python src/2_train.py
    python src/3_evaluate.py
    ```
3.  **Archive:** Results are automatically routed to `results/[EXPERIMENT_NAME]/generated_tabs`.

## 📈 Final Performance
The model was validated using the **GuitarSet** dataset (Player 05 hold-out).

| Method | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- |
| Raw CNN Output | 0.XXXX | 0.XXXX | 0.XXXX |
| **Viterbi (Physics Aware)** | **0.XXXX** | **0.XXXX** | **0.XXXX** |
*(Note: Replace XXXX with values from results/metrics/final_results.csv before archival)*

---
**Archival Date:** Friday, April 3, 2026  
**Next Project:** Solo Neuroscience (EEG Signal Processing)
```
