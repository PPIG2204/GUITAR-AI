import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Point to your newly preprocessed synthetic data
data_path = "../../processed_data/synthetic/*.npz"
files = glob.glob(data_path)

if not files:
    print(f"❌ No files found in {data_path}. Check your preprocessing output.")
    exit()

# Initialize distribution (6 strings, 21 frets)
# Note: GuitarSet typically uses 21 frets (0-20)
dist = np.zeros((6, 21)) 

print(f"📊 Analyzing {len(files)} synthetic chords...")

for f in tqdm(files):
    data = np.load(f)
    labels = data['labels'] # Shape: (Time, 6, 21)
    
    # Sum across the time axis (axis 0) 
    # and add to the global distribution
    dist += np.sum(labels, axis=0)

# Plotting the result
plt.figure(figsize=(12, 6))
plt.imshow(dist, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Frame Count')
plt.title("Synthetic Fretboard Distribution (Processed NPZ)")
plt.xlabel("Fret Number")
plt.ylabel("String (0=High E, 5=Low E)")
plt.xticks(range(21))
plt.yticks(range(6))
plt.show()