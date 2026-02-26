import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# Modified data_trend for raw atoms
atoms = glob.glob("../../data/atoms/*.wav")
dist = np.zeros((6, 25)) # Up to fret 24

for a in atoms:
    parts = os.path.basename(a).split('_')
    s = int(parts[0].replace('str', ''))
    f = int(parts[1].replace('fr', ''))
    dist[s, f] += 1

plt.imshow(dist, aspect='auto', origin='lower', cmap='viridis')
plt.title("Harvested Note Distribution (Atoms)")
plt.xlabel("Fret")
plt.ylabel("String")
plt.show()