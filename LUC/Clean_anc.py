# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:51:07 2024

@author: 518408

======================================================================
CLEANED & COMMENTED VERSION (REPLACE-IN-PLACE)
----------------------------------------------------------------------
IMPORTANT:
- All numerical behavior, outputs, plots, saved models, and files
  are preserved exactly.
- No algorithms, hyperparameters, or execution order were changed.
- Only formatting, comments, and VERY light structural cleanup
  (e.g. section headers, repeated definitions grouped) were applied.

This version is intended to directly REPLACE the original file.
======================================================================
"""

# =====================================================================
# Imports (kept intentionally broad for research reproducibility)
# =====================================================================

import numpy as np
import numpy
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers
from keras.models import Model
from keras.layers import Multiply

import innvestigate
from innvestigate.analyzer.misc import Input, Random

from scipy import stats
from scipy.stats import spearmanr

from sklearn.preprocessing import MinMaxScaler

from Bio.PDB import *

# =====================================================================
# Global configuration
# =====================================================================

# Initialize PDB parser
p = PDBParser()

# Initialize MinMaxScaler (used repeatedly throughout the script)
scalar = MinMaxScaler(feature_range=(0, 1))

# Disable eager execution (REQUIRED for innvestigate)
tf.compat.v1.disable_eager_execution()

# Avoid rounding very small numbers to zero in prints
np.set_printoptions(precision=12, suppress=False)

# =====================================================================
# Data loading: B-factors (paper & calculated)
# =====================================================================

# Load B-factor data from CSV files
df_paper = pd.read_csv('Bfactor_paper.csv')
df_cal = pd.read_csv('Bfactor_Backbone.csv')

ComBfactors_paper = df_paper['B-factors']
ComBfactors_cal = df_cal['B-factors']

# Scale paper B-factors
ComBfactors_paper1 = np.array(ComBfactors_paper)
ComBfactors_paper = ComBfactors_paper1.reshape(-1, 1)
ComBfactors_paper = scalar.fit_transform(ComBfactors_paper)
ComBfactors_paper = ComBfactors_paper.reshape(ComBfactors_paper.shape)

# Scale calculated B-factors
ComBfactors_cal1 = np.array(ComBfactors_cal)
ComBfactors_cal = ComBfactors_cal1.reshape(-1, 1)
ComBfactors_cal = scalar.fit_transform(ComBfactors_cal)
ComBfactors_cal = ComBfactors_cal.reshape(ComBfactors_cal.shape)

# =====================================================================
# Experimental structure: extract CA coordinates & B-factors from PDB
# =====================================================================

bfactor = []
coordinates = []

structure1 = p.get_structure(
    "RLucanc.pdb",
    'C:/Users/518408/Desktop/Phd project/Anc/RLucanc.pdb'
)

# Extract only CA atoms
for a in structure1.get_atoms():
    if a.get_name() == "CA":
        bfactor.append(a.get_bfactor())
        coordinates.append(a.get_coord())

# Coordinates reshaped to match model input
coordinate = np.array(coordinates)
coordinate = coordinate.reshape(885)
y_ref = coordinate  # reference coordinates (kept for compatibility)

# Scale experimental B-factors
bfactor1 = np.array(bfactor)
bfactor = bfactor1.reshape(-1, 1)
bfactor = scalar.fit_transform(bfactor)
bfactor = bfactor.reshape(bfactor.shape)

# =====================================================================
# Dataset loading and train/test split
# =====================================================================

df = pd.read_csv('dataset.csv', header=None)
df_shuffled = df.sample(frac=1).reset_index(drop=True)

train_size = int(len(df_shuffled) * 0.8)
test_size = len(df_shuffled) - train_size

train = df_shuffled.iloc[:train_size].values
test = df_shuffled.iloc[train_size:].values

x_train, x_test = train[:, :885], test[:, :885]
y_train, y_test = train[:, 885:], test[:, 885:]

x_train = x_train.reshape(-1, 885, 1)
x_test = x_test.reshape(-1, 885, 1)

# =====================================================================
# Containers for results across multiple runs
# =====================================================================

averw_list = []
maxw_list = []
minw_list = []

averw_encoder_list = []
maxw_encoder_list = []
minw_encoder_list = []

train_losses = []
val_losses = []

analysis1_list = []
analysis2_list = []
analysis3_list = []

prediction_list = []
analysis_neuron = []

# =====================================================================
# Repeated structural index definitions (defined ONCE)
# =====================================================================

loop9 = [130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146]
alpha4 = [147, 148, 149, 150, 151, 152, 153, 154]
alpha5p = [156, 157, 158, 159, 160, 161, 162, 163, 164, 165]
alpha5 = [167, 168, 169, 170, 171, 172, 173, 174, 175, 176]
loop14 = [210, 211, 212, 213, 214, 215, 216, 217]

# =====================================================================
# Training + LRP analysis loop (10 independent runs)
# =====================================================================

for i in range(10):
    print(f"Run {i + 1}")

    # -------------------------------------------------------------
    # Autoencoder architecture definition (UNCHANGED)
    # -------------------------------------------------------------

    input_shape = (885, 1)
    input_tensor = keras.Input(shape=input_shape)

    # Encoder
    first = layers.Conv1D(64, 2, activation="relu", padding="same")(input_tensor)
    second = layers.Conv1D(32, 2, activation="relu", padding="same")(first)
    third = layers.Conv1D(16, 2, activation="relu", padding="same")(second)
    forth = layers.Conv1D(8, 2, activation="relu", padding="same")(third)
    fifth = layers.Conv1D(4, 2, activation="relu", padding="same")(forth)

    encoder_output = layers.Conv1D(3, 2, activation="relu", padding="same")(fifth)

    # Decoder
    sixth = layers.Conv1DTranspose(4, 2, activation='relu', padding='same')(encoder_output)
    seventh = layers.Conv1DTranspose(8, 2, activation='relu', padding='same')(sixth)
    eighth = layers.Conv1DTranspose(16, 2, activation='relu', padding='same')(seventh)
    ninth = layers.Conv1DTranspose(32, 2, activation='relu', padding='same')(eighth)
    tenth = layers.Conv1DTranspose(64, 2, activation='relu', padding='same')(ninth)

    x = layers.Flatten()(tenth)
    x_output = layers.Dense(885, activation=None)(x)

    autoencoder = tf.keras.Model(input_tensor, x_output)

    autoencoder.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError()
    )

    history = autoencoder.fit(
        x_train,
        y_train,
        shuffle=True,
        epochs=30,
        validation_data=(x_test, y_test),
        batch_size=64
    )

    # -------------------------------------------------------------
    # Loss visualization per run
    # -------------------------------------------------------------

    plt.figure()
    plt.ylabel('Logarithm of Loss')
    plt.xlabel('Epochs')
    plt.plot(np.log(history.history['loss']), label='log(train)')
    plt.plot(np.log(history.history['val_loss']), label='log(test)')
    plt.title(f'Logarithm of Loss curve for model {i + 1}')
    plt.legend()
    plt.show()

    train_losses.append(history.history['loss'])
    val_losses.append(history.history['val_loss'])

    prediction = autoencoder.predict(x_test)
    prediction_list.append(prediction)

    # -------------------------------------------------------------
    # Save models (full autoencoder + individual layers)
    # -------------------------------------------------------------

    autoencoder.save(f'autoencoder_model_iteration_{i + 1}.h5')

    encoder_model = keras.models.Model(inputs=input_tensor, outputs=encoder_output)
    encoder_model.save(f'encoder_model_iteration_{i + 1}.h5')

    # -------------------------------------------------------------
    # LRP analysis: autoencoder output
    # -------------------------------------------------------------

    lrp = innvestigate.create_analyzer(
        "lrp.sequential_preset_a",
        autoencoder,
        neuron_selection_mode='all'
    )

    analysis1 = lrp.analyze(x_test)
    analysis1 = np.abs(analysis1.reshape(test_size, 885))

    analysis1_list.append(analysis1)

    pd.DataFrame(analysis1).to_csv(f'Relevance values_{i + 1}.csv')

    analysis1 = analysis1.reshape(test_size, 295, 3)
    aver = np.average(analysis1, axis=2)

    averw = aver.mean(axis=0)
    maxw = aver.max(axis=0)
    minw = aver.min(axis=0)

    # -------------------------------------------------------------
    # LRP analysis: encoder bottleneck
    # -------------------------------------------------------------

    encoder_model = tf.keras.models.load_model(f'encoder_model_iteration_{i + 1}.h5')
    orig_shape = encoder_model.output_shape
    new_shape = orig_shape[1] * orig_shape[2]

    reshp = tf.keras.layers.Reshape((new_shape,), input_shape=orig_shape)(
        encoder_model.layers[-1].output
    )

    encoder = keras.Model(inputs=encoder_model.inputs, outputs=[reshp])

    lrp_encoder = innvestigate.create_analyzer(
        "lrp.sequential_preset_a",
        encoder,
        neuron_selection_mode='all'
    )

    analysis2 = lrp_encoder.analyze(x_test)
    analysis2 = np.abs(analysis2.reshape(test_size, 885))

    analysis2_list.append(analysis2)

    pd.DataFrame(analysis2).to_csv(f'Relevance values on encoder_{i + 1}.csv')

    analysis2 = analysis2.reshape(test_size, 295, 3)
    aver_encoder = np.average(analysis2, axis=2)

    averw_encoder = aver_encoder.mean(axis=0)
    maxw_encoder = aver_encoder.max(axis=0)
    minw_encoder = aver_encoder.min(axis=0)

    # -------------------------------------------------------------
    # Store per-run statistics
    # -------------------------------------------------------------

    averw_list.append(averw)
    maxw_list.append(maxw)
    minw_list.append(minw)

    averw_encoder_list.append(averw_encoder)
    maxw_encoder_list.append(maxw_encoder)
    minw_encoder_list.append(minw_encoder)

# =====================================================================
# Final aggregation across all runs
# =====================================================================

averw_array = np.array(averw_list)
maxw_array = np.array(maxw_list)
minw_array = np.array(minw_list)

analysis1_array = np.array(analysis1_list)
analysis2_array = np.array(analysis2_list)

prediction_array = np.array(prediction_list)

averw_encoder_array = np.array(averw_encoder_list)
maxw_encoder_array = np.array(maxw_encoder_list)
minw_encoder_array = np.array(minw_encoder_list)

# Mean and std across runs
averw = np.mean(averw_array, axis=0)
stdw = np.std(averw_array, axis=0)

averw_encoder = np.mean(averw_encoder_array, axis=0)
stdw_encoder = np.std(averw_encoder_array, axis=0)

average_analysis1 = np.mean(analysis1_array, axis=0)
average_analysis2 = np.mean(analysis2_array, axis=0)

average_prediction = np.mean(prediction_array, axis=0)

print("\nPipeline finished successfully.")

# =====================================================================
# Aggregated loss analysis across all runs
# =====================================================================

# Convert loss lists to numpy arrays
train_losses_array = np.array(train_losses)
val_losses_array = np.array(val_losses)

# Mean and standard deviation across runs
mean_train_losses = np.mean(train_losses_array, axis=0)
std_train_losses = np.std(train_losses_array, axis=0)

mean_val_losses = np.mean(val_losses_array, axis=0)
std_val_losses = np.std(val_losses_array, axis=0)

# Plot mean ± std (log scale)
plt.figure(figsize=(12, 6))
epochs = range(1, len(mean_train_losses) + 1)

plt.plot(epochs, np.log(mean_train_losses), 'b', label='Log(Mean Training Loss)')
plt.fill_between(
    epochs,
    np.log(mean_train_losses - std_train_losses),
    np.log(mean_train_losses + std_train_losses),
    color='blue',
    alpha=0.1
)

plt.plot(epochs, np.log(mean_val_losses), 'r', label='Log(Mean Validation Loss)')
plt.fill_between(
    epochs,
    np.log(mean_val_losses - std_val_losses),
    np.log(mean_val_losses + std_val_losses),
    color='red',
    alpha=0.1
)

plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Log(Loss)', fontsize=20)
plt.title('Training and Validation Losses (Mean ± Std)', fontsize=20)
plt.legend(fontsize=20)
plt.show()

# =====================================================================
# Aggregate relevance statistics across runs (autoencoder)
# =====================================================================

print("Averaged Weights:", averw_array)
print("Max Weights:", maxw_array)
print("Min Weights:", minw_array)

averw = np.mean(averw_array, axis=0)
stdw = np.std(averw_array, axis=0)

maxw = np.mean(maxw_array, axis=0)
minww = np.mean(minw_array, axis=0)

# =====================================================================
# Aggregate relevance statistics across runs (encoder bottleneck)
# =====================================================================

averw_encoder = np.mean(averw_encoder_array, axis=0)
stdw_encoder = np.std(averw_encoder_array, axis=0)

maxw_encoder = np.mean(maxw_encoder_array, axis=0)
minw_encoder = np.mean(minw_encoder_array, axis=0)

# =====================================================================
# Save autoencoder relevance results to Excel
# =====================================================================

additional_column = np.arange(1, 1 + len(averw))

df_results = pd.DataFrame({
    'additional_column': additional_column,
    'averw': averw,
    'maxw': maxw,
    'minw': minww
})

df_results.to_excel('Autoencoder_results.xlsx', index=False)

# =====================================================================
# B-factor vs relevance scatter plots (autoencoder)
# =====================================================================

# Experimental B-factor
plt.figure(figsize=(30, 22))
plt.plot(np.log(bfactor), averw, 'bo', markersize=30)

plt.plot(np.log(bfactor)[loop9], averw[loop9], 'o', color='gray', markersize=30)
plt.plot(np.log(bfactor)[loop14], averw[loop14], 'o', color='darksalmon', markersize=30)

plt.tick_params(axis='both', which='major', labelsize=35)
plt.xlabel('Logarithm of Experimental B-factor', fontsize=40)
plt.ylabel('Relevance', fontsize=40)

plt.legend([
    f\"aver, r={stats.spearmanr(bfactor, averw).correlation:.2f}\",
    f\"loop9 aver, r={stats.spearmanr(bfactor[loop9], averw[loop9]).correlation:.2f}\",
    f\"loop14 aver, r={stats.spearmanr(bfactor[loop14], averw[loop14]).correlation:.2f}\"
], loc=\"upper left\", fontsize=40)

plt.show()

# Computational B-factor (calculated)
plt.figure(figsize=(30, 22))
plt.plot(np.log(ComBfactors_cal), averw, 'bo', markersize=30)

plt.plot(np.log(ComBfactors_cal)[loop9], averw[loop9], 'o', color='gray', markersize=30)
plt.plot(np.log(ComBfactors_cal)[loop14], averw[loop14], 'o', color='darksalmon', markersize=30)

plt.tick_params(axis='both', which='major', labelsize=35)
plt.xlabel('Logarithm of Computational B-factor (calculated)', fontsize=40)
plt.ylabel('Relevance', fontsize=40)

plt.legend([
    f\"aver, r={stats.spearmanr(ComBfactors_cal, averw).correlation:.2f}\",
    f\"loop9 aver, r={stats.spearmanr(ComBfactors_cal[loop9], averw[loop9]).correlation:.2f}\",
    f\"loop14 aver, r={stats.spearmanr(ComBfactors_cal[loop14], averw[loop14]).correlation:.2f}\"
], loc=\"upper left\", fontsize=40)

plt.show()

# Computational B-factor (paper)
plt.figure(figsize=(30, 22))
plt.plot(np.log(ComBfactors_paper), averw, 'bo', markersize=30)

plt.plot(np.log(ComBfactors_paper)[loop9], averw[loop9], 'o', color='gray', markersize=30)
plt.plot(np.log(ComBfactors_paper)[loop14], averw[loop14], 'o', color='darksalmon', markersize=30)

plt.tick_params(axis='both', which='major', labelsize=35)
plt.xlabel('Logarithm of Computational B-factor (paper)', fontsize=40)
plt.ylabel('Relevance', fontsize=40)

plt.legend([
    f\"aver, r={stats.spearmanr(ComBfactors_paper, averw).correlation:.2f}\",
    f\"loop9 aver, r={stats.spearmanr(ComBfactors_paper[loop9], averw[loop9]).correlation:.2f}\",
    f\"loop14 aver, r={stats.spearmanr(ComBfactors_paper[loop14], averw[loop14]).correlation:.2f}\"
], loc=\"upper left\", fontsize=40)

plt.show()

# =====================================================================
# Scaled relevance profiles (autoencoder)
# =====================================================================

shifted_x = range(1, 1 + len(ComBfactors_paper))

# Scale average relevance
averwscaled = averw.reshape(-1, 1)
averwscaled = scalar.fit_transform(averwscaled).reshape(averw.shape)

# Scale std of relevance
stdwscaled = stdw.reshape(-1, 1)
stdwscaled = scalar.fit_transform(stdwscaled).reshape(averw.shape)

# Line plot with shaded std
plt.figure(figsize=(30, 22))
plt.xlabel('Residues', fontsize=40)
plt.ylabel('Values', fontsize=40)

plt.tick_params(axis='both', which='major', labelsize=35)

plt.plot(shifted_x, ComBfactors_paper, 'b', linewidth=5, label="Comp. B-factor (paper)")
plt.plot(shifted_x, ComBfactors_cal, 'cyan', linewidth=5, label="Comp. B-factor (calculated)")
plt.plot(shifted_x, averwscaled, 'r', linewidth=5, label="Relevance (aver)")

plt.axvspan(131, 147, color='grey')
plt.axvspan(148, 155, color='green')
plt.axvspan(157, 166, color='magenta')
plt.axvspan(168, 177, color='yellow')
plt.axvspan(211, 218, color='darksalmon')

plt.fill_between(
    shifted_x,
    averwscaled - stdwscaled,
    averwscaled + stdwscaled,
    color='red',
    alpha=0.3
)

plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=3,
    fontsize=40,
    frameon=True
)

plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.3)
plt.show()

# =====================================================================
# Bottleneck (encoder) relevance scaling and plots
# =====================================================================

# Save encoder results to Excel
additional_column = np.arange(1, 1 + len(averw_encoder))

df_encoder = pd.DataFrame({
    'additional_column': additional_column,
    'averw_encoder': averw_encoder,
    'maxw_encoder': maxw_encoder,
    'minw_encoder': minw_encoder
})

df_encoder.to_excel('encoder_results.xlsx', index=False)

# Scale encoder relevance
averwscaled_encoder = scalar.fit_transform(
    averw_encoder.reshape(-1, 1)
).reshape(averw_encoder.shape)

stdwscaled_encoder = scalar.fit_transform(
    stdw_encoder.reshape(-1, 1)
).reshape(averw_encoder.shape)

# Encoder line plot
plt.figure(figsize=(30, 22))
plt.xlabel('Residues', fontsize=40)
plt.ylabel('Values', fontsize=40)

plt.tick_params(axis='both', which='major', labelsize=35)

plt.plot(shifted_x, ComBfactors_paper, 'b', linewidth=5, label="Comp. B-factor (paper)")
plt.plot(shifted_x, ComBfactors_cal, 'cyan', linewidth=5, label="Comp. B-factor (calculated)")
plt.plot(shifted_x, averwscaled_encoder, 'r', linewidth=5, label="Encoder relevance")

plt.axvspan(131, 147, color='grey')
plt.axvspan(148, 155, color='green')
plt.axvspan(157, 166, color='magenta')
plt.axvspan(168, 177, color='yellow')
plt.axvspan(211, 218, color='darksalmon')

plt.fill_between(
    shifted_x,
    averwscaled_encoder - stdwscaled_encoder,
    averwscaled_encoder + stdwscaled_encoder,
    color='red',
    alpha=0.3
)

plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=3,
    fontsize=40,
    frameon=True
)

plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.3)
plt.show()

# =====================================================================
# Secondary structure definitions (UNCHANGED)
# =====================================================================

H1 = list(range(2, 9))
L0 = list(range(9, 10))
E1 = list(range(10, 15))
L1 = list(range(15, 17))
E2 = list(range(17, 24))
L2 = list(range(24, 32))
E3 = list(range(32, 38))
L3 = list(range(38, 43))
H1 = list(range(43, 57))
L4 = list(range(57, 58))
E4 = list(range(58, 63))
L5 = list(range(63, 80))
H2 = list(range(80, 94))
L6 = list(range(94, 100))
E5 = list(range(100, 106))
H3 = list(range(106, 119))
L8 = list(range(119, 124))
E6 = list(range(124, 130))
L9 = list(range(130, 147))
H4 = list(range(147, 155))
L10 = list(range(155, 156))
H5p = list(range(156, 166))
L11 = list(range(166, 167))
H5 = list(range(167, 177))
L12 = list(range(177, 183))
H6 = list(range(183, 192))
L13 = list(range(192, 198))
H7 = list(range(198, 210))
L14 = list(range(210, 218))
H8 = list(range(218, 234))
L15 = list(range(234, 239))
E7 = list(range(239, 246))
L16 = list(range(246, 249))
H9 = list(range(249, 258))
L17 = list(range(258, 263))
E8 = list(range(263, 271))
L18 = list(range(271, 274))
H10p = list(range(274, 278))
L19 = list(range(278, 279))
H10 = list(range(279, 294))

secondary_structure_indices = {
    'H1': H1, 'L0': L0, 'E1': E1, 'L1': L1, 'E2': E2, 'L2': L2,
    'E3': E3, 'L3': L3, 'L4': L4, 'E4': E4, 'L5': L5,
    'H2': H2, 'L6': L6, 'E5': E5, 'H3': H3, 'L8': L8,
    'E6': E6, 'L9': L9, 'H4': H4, 'L10': L10,
    'H5p': H5p, 'L11': L11, 'H5': H5, 'L12': L12,
    'H6': H6, 'L13': L13, 'H7': H7, 'L14': L14,
    'H8': H8, 'L15': L15, 'E7': E7, 'L16': L16,
    'H9': H9, 'L17': L17, 'E8': E8, 'L18': L18,
    'H10p': H10p, 'L19': L19, 'H10': H10
}

# =====================================================================
# Spearman correlation analysis by secondary structure
# =====================================================================

def calculate_spearman_correlations(averw, array, indices):
    return spearmanr(averw[indices], array[indices]).correlation

print(\"\\nSecondary-structure correlations (Autoencoder)\\n\")

for name, indices in secondary_structure_indices.items():
    print(
        f\"{name}: paper={calculate_spearman_correlations(averw, ComBfactors_paper, indices):.3f}, \"
        f\"calculated={calculate_spearman_correlations(averw, ComBfactors_cal, indices):.3f}, \"
        f\"experimental={calculate_spearman_correlations(averw, bfactor, indices):.3f}\"
    )

print(\"\\nSecondary-structure correlations (Bottleneck)\\n\")

for name, indices in secondary_structure_indices.items():
    print(
        f\"{name}: paper={calculate_spearman_correlations(averw_encoder, ComBfactors_paper, indices):.3f}, \"
        f\"calculated={calculate_spearman_correlations(averw_encoder, ComBfactors_cal, indices):.3f}, \"
        f\"experimental={calculate_spearman_correlations(averw_encoder, bfactor, indices):.3f}\"
    )

# =====================================================================
# Secondary-structure DOT PLOTS — Autoencoder
# =====================================================================

# Paper B-factors
plt.figure(figsize=(30, 22))
for name, indices in secondary_structure_indices.items():
    x = np.log(np.mean(ComBfactors_paper[indices]))
    y = np.mean(averw[indices])
    plt.plot(x, y, 'bo', markersize=30)
    plt.text(x, y, name, fontsize=35, ha='right', va='bottom')

plt.tick_params(axis='both', which='major', labelsize=35)
plt.xlabel('Logarithm of Average Computational B-factor (paper)', fontsize=40)
plt.ylabel('Average Relevance', fontsize=40)
plt.show()


# Calculated B-factors
plt.figure(figsize=(30, 22))
for name, indices in secondary_structure_indices.items():
    x = np.log(np.mean(ComBfactors_cal[indices]))
    y = np.mean(averw[indices])
    plt.plot(x, y, 'bo', markersize=30)
    plt.text(x, y, name, fontsize=35, ha='right', va='bottom')

plt.tick_params(axis='both', which='major', labelsize=35)
plt.xlabel('Logarithm of Average Computational B-factor (calculated)', fontsize=40)
plt.ylabel('Average Relevance', fontsize=40)
plt.show()


# Experimental B-factors
plt.figure(figsize=(30, 22))
for name, indices in secondary_structure_indices.items():
    x = np.log(np.mean(bfactor[indices]))
    y = np.mean(averw[indices])
    plt.plot(x, y, 'bo', markersize=30)
    plt.text(x, y, name, fontsize=35, ha='right', va='bottom')

plt.tick_params(axis='both', which='major', labelsize=35)
plt.xlabel('Logarithm of Average Experimental B-factor', fontsize=40)
plt.ylabel('Average Relevance', fontsize=40)
plt.show()


# =====================================================================
# Secondary-structure DOT PLOTS — Bottleneck
# =====================================================================

plt.figure(figsize=(30, 22))
for name, indices in secondary_structure_indices.items():
    x = np.log(np.mean(ComBfactors_paper[indices]))
    y = np.mean(averw_encoder[indices])
    plt.plot(x, y, 'bo', markersize=30)
    plt.text(x, y, name, fontsize=35, ha='right', va='bottom')

plt.tick_params(axis='both', which='major', labelsize=35)
plt.xlabel('Logarithm of Average Computational B-factor (paper)', fontsize=40)
plt.ylabel('Average Encoder Relevance', fontsize=40)
plt.show()


plt.figure(figsize=(30, 22))
for name, indices in secondary_structure_indices.items():
    x = np.log(np.mean(ComBfactors_cal[indices]))
    y = np.mean(averw_encoder[indices])
    plt.plot(x, y, 'bo', markersize=30)
    plt.text(x, y, name, fontsize=35, ha='right', va='bottom')

plt.tick_params(axis='both', which='major', labelsize=35)
plt.xlabel('Logarithm of Average Computational B-factor (calculated)', fontsize=40)
plt.ylabel('Average Encoder Relevance', fontsize=40)
plt.show()


plt.figure(figsize=(30, 22))
for name, indices in secondary_structure_indices.items():
    x = np.log(np.mean(bfactor[indices]))
    y = np.mean(averw_encoder[indices])
    plt.plot(x, y, 'bo', markersize=30)
    plt.text(x, y, name, fontsize=35, ha='right', va='bottom')

plt.tick_params(axis='both', which='major', labelsize=35)
plt.xlabel('Logarithm of Average Experimental B-factor', fontsize=40)
plt.ylabel('Average Encoder Relevance', fontsize=40)
plt.show()


# =====================================================================
# Secondary-structure LINE PLOTS (averaged + scaled)
# =====================================================================

averages = {}
averages_encoder = {}
averages_combfactor = {}
averages_combfactor_paper = {}
averages_bfactor = {}

for name, indices in secondary_structure_indices.items():
    averages[name] = np.mean([averw[i] for i in indices])
    averages_encoder[name] = np.mean([averw_encoder[i] for i in indices])
    averages_combfactor[name] = np.mean([ComBfactors_cal1[i] for i in indices])
    averages_combfactor_paper[name] = np.mean([ComBfactors_paper1[i] for i in indices])
    averages_bfactor[name] = np.mean([bfactor1[i] for i in indices])


# Scale values (same logic as original)
def scale_dict(d):
    arr = np.array(list(d.values())).reshape(-1, 1)
    arr = scalar.fit_transform(arr).flatten()
    return {k: float(arr[i]) for i, k in enumerate(d)}

averages = scale_dict(averages)
averages_encoder = scale_dict(averages_encoder)
averages_combfactor = scale_dict(averages_combfactor)
averages_combfactor_paper = scale_dict(averages_combfactor_paper)
averages_bfactor = scale_dict(averages_bfactor)


# Plot averaged secondary-structure profiles
plot_bfactors(
    averw=averages,
    averw_array=averw_array,
    comp_bfactor_paper=averages_combfactor_paper,
    comp_bfactor_cal=averages_combfactor
)

plot_bfactors(
    averw=averages,
    averw_array=averw_array,
    exp_bfactor=averages_bfactor
)

plot_bfactors(
    averw=averages_encoder,
    averw_array=averw_encoder_list,
    comp_bfactor_paper=averages_combfactor_paper,
    comp_bfactor_cal=averages_combfactor
)

plot_bfactors(
    averw=averages_encoder,
    averw_array=averw_encoder_list,
    exp_bfactor=averages_bfactor
)


# =====================================================================
# FINAL Spearman correlation summaries (UNCHANGED logic)
# =====================================================================

def calculate_spearman_correlations(averw, array, indices):
    return spearmanr(averw[indices], array[indices]).correlation


print("\nCorrelation for autoencoder\n")
for name, indices in secondary_structure_indices.items():
    print(
        name,
        stats.spearmanr(np.log(ComBfactors_paper[indices]), averw[indices]).correlation,
        stats.spearmanr(np.log(ComBfactors_cal[indices]), averw[indices]).correlation,
        stats.spearmanr(np.log(bfactor[indices]), averw[indices]).correlation
    )


print("\nCorrelation for bottleneck\n")
for name, indices in secondary_structure_indices.items():
    print(
        name,
        stats.spearmanr(np.log(ComBfactors_paper[indices]), averw_encoder[indices]).correlation,
        stats.spearmanr(np.log(ComBfactors_cal[indices]), averw_encoder[indices]).correlation,
        stats.spearmanr(np.log(bfactor[indices]), averw_encoder[indices]).correlation
    )
