import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import time

# Relative path from src/scripts/ to the npz files
path_to_data = os.path.join("processed_data", "train", "*.npz")
files = glob.glob(path_to_data)

print(f"🔍 Searching in: {os.path.abspath(path_to_data)}")
print(f"📦 Total files found: {len(files)}")

if len(files) == 0:
    print("❌ Still zero files. Try using an absolute path or check if you are running from the /scripts/ folder.")
    exit()

# Initialize [6 strings, 21 frets + 1 for open]
all_fret_dist = np.zeros((6, 22))

# 1. Align the dimensions (6 strings, 21 frets)
# Use 21 because your data has shape (6, 21)
all_fret_dist = np.zeros((6, 21)) 

start_time = time.time()

for i, f in enumerate(files):
    # Progress Reporter
    if i % 1000 == 0:
        elapsed = time.time() - start_time
        print(f"🔄 Processed {i}/{len(files)} files... ({elapsed:.1f}s)")
    
    data = np.load(f)

    labels = data['labels']
    
    # Sum across time (axis 0) to get a (6, 21) result
    # We remove the [:, :22] slicing which was causing the mismatch
    if labels.ndim == 3:
        all_fret_dist += np.sum(labels, axis=0) 
    else:
        all_fret_dist += labels

# 2. Plotting the Heatmap with Log Scale 
# (Better for seeing rare high-fret notes)
plt.figure(figsize=(14, 6))
plt.imshow(all_fret_dist, aspect='auto', cmap='magma', origin='lower')
plt.colorbar(label='Frequency of Occurrence')
plt.title("Physical Note Distribution: Guitar Fretboard Heatmap")
plt.xlabel("Fret Number")
plt.ylabel("String (0=High E, 5=Low E)")
plt.xticks(range(21))
plt.yticks(range(6))
plt.show()