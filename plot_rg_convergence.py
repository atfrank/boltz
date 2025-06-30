#!/usr/bin/env python3
"""
Simple plotting script to visualize Rg convergence.
"""

import matplotlib.pyplot as plt
import numpy as np

# Read the data
data = []
with open('rg_trajectory_analysis.txt', 'r') as f:
    next(f)  # Skip header
    for line in f:
        frame, rg = line.strip().split(',')
        data.append((int(frame), float(rg)))

frames = [d[0] for d in data]
rg_values = [d[1] for d in data]

# Create the plot
plt.figure(figsize=(12, 8))

# Main plot
plt.subplot(2, 1, 1)
plt.plot(frames, rg_values, 'b-', linewidth=2, label='Rg')
plt.xlabel('Diffusion Step')
plt.ylabel('Radius of Gyration (Å)')
plt.title('Rg Convergence During Diffusion Process')
plt.grid(True, alpha=0.3)
plt.legend()

# Log scale plot to better see the convergence
plt.subplot(2, 1, 2)
plt.semilogy(frames, rg_values, 'r-', linewidth=2, label='Rg (log scale)')
plt.xlabel('Diffusion Step')
plt.ylabel('Radius of Gyration (Å) - Log Scale')
plt.title('Rg Convergence (Log Scale)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('rg_convergence.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved as rg_convergence.png")