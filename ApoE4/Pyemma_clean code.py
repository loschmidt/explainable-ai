# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 17:01:59 2023

Author: 518408

Description
-----------
Compute frame-to-frame differences of minimum inter-residue distances
from molecular dynamics trajectories using PyEMMA.

This script:
- loads a reference PDB
- computes residue-wise minimum distances
- calculates differences between consecutive frames
- merges results from multiple .short.xtc trajectories
- writes the final dataset to a CSV file

This file is a CLEAN conversion from a Jupyter notebook:
- NO logic changes
- NO numerical changes
- ONLY formatting, comments, and structure cleanup
"""

# =====================================================================
# Imports
# =====================================================================

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

import pyemma.coordinates as coor


# =====================================================================
# PyEMMA featurizer setup
# =====================================================================

# Create a featurizer using the reference structure
feat = coor.featurizer("output.strip.pdb")

# Add minimum inter-residue distance feature
# NOTE: This considers all residue pairs and can be computationally expensive
feat.add_residue_mindist()


# =====================================================================
# Containers
# =====================================================================

# Will store arrays of shape (frames-1, n_distances) for each trajectory
residue_mindist_array = []


# =====================================================================
# Core computation
# =====================================================================

def compute_residue_mindist_for_file(file_path):
    """
    Compute frame-to-frame differences of residue minimum distances
    for a single trajectory file and append the result to the global list.
    """
    try:
        # Load trajectory with PyEMMA
        source = coor.source(file_path, features=feat)
        data = source.get_output()

        # data[0] has shape: (n_frames, n_distances)
        residue_mindist = data[0]

        # Compute difference between consecutive frames
        diff = residue_mindist[1:] - residue_mindist[:-1]

        # Store results
        residue_mindist_array.append(diff)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


def apply_residue_mindist(directory):
    """
    Find all '.short.xtc' files in a directory and apply
    residue minimum distance computation to each.
    """
    short_xtc_files = [
        file for file in os.listdir(directory)
        if file.endswith(".short.xtc")
    ]

    if not short_xtc_files:
        print("No '.short.xtc' files found in the directory.")
        return

    for file in short_xtc_files:
        file_path = os.path.join(directory, file)
        print(f"Processing file: {file_path}")
        compute_residue_mindist_for_file(file_path)


# =====================================================================
# Execution
# =====================================================================

# Directory containing the .short.xtc files
directory_path = "./"

# Run computation over all trajectories
apply_residue_mindist(directory_path)

# Merge all trajectory results into a single array
merged_list = [inner for outer in residue_mindist_array for inner in outer]
merged_list_array = np.array(merged_list)


# =====================================================================
# Save output
# =====================================================================

output_csv = "data_train_with_3spa_difference.csv"

with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(merged_list_array)

print(f"Saved merged residue distance differences to {output_csv}")


# =====================================================================
# Optional visualization (commented out — preserved from notebook)
# =====================================================================

# The following block was present in the original notebook and is kept
# here as comments to preserve intent and reproducibility.
#
# It reconstructs square distance matrices and plots heatmaps.
#
# ---------------------------------------------------------------------
#
# num_residues = 135
#
# for i in range(len(merged_list_array)):
#     distances = merged_list_array[i]
#     square_matrix = np.zeros((num_residues, num_residues))
#
#     for residue in range(num_residues):
#         start_index = (residue * num_residues) - (((residue - 1) * residue) // 2)
#         end_index = start_index + num_residues - residue - 1
#         square_matrix[residue, residue + 1:] = distances[start_index:end_index]
#
#     plt.figure(figsize=(8, 6))
#     plt.imshow(square_matrix, cmap="viridis", interpolation="nearest")
#     plt.colorbar(label="Distance")
#     plt.title(f"Heatmap for Row {i + 1}")
#     plt.xlabel("Residue Index")
#     plt.ylabel("Residue Index")
#     plt.show()
