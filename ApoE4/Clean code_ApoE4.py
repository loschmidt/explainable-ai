# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:39:13 2024

Author: 518408

Description
-----------
Binary classification of protein distance-matrix differences using a CNN,
with K-fold cross-validation and Layer-wise Relevance Propagation (LRP)
analysis via innvestigate.
"""

# =====================================================================
# Imports
# =====================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow.keras import layers, Input, Model, regularizers
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import innvestigate
from scipy.stats import spearmanr


# =====================================================================
# Data loading
# =====================================================================

# Load train and test CSV files
df_train = pd.read_csv("difference_of_distance_train.csv", header=None)
df_test = pd.read_csv("difference_of_distance_test.csv", header=None)

# Subsample training data (every third row)
df_train = df_train[df_train.index % 3 == 0].reset_index(drop=True)

# Features: first 9180 columns
X_train = df_train.iloc[:, :9180]
X_test = df_test.iloc[:, :9180]

# Labels: last column
y_train = df_train.iloc[:, -1]
y_test = df_test.iloc[:, -1]


# =====================================================================
# Utility: reconstruct distance matrix (excluding neighbors)
# =====================================================================

def recreate_distance_matrix_exclude_neighbors(distance_vector, num_residues=138):
    """
    Reconstruct a symmetric distance matrix from a flattened vector,
    excluding diagonal and nearest neighbors (±1, ±2 residues).
    """
    distance_matrix = np.full((num_residues, num_residues), np.inf)
    np.fill_diagonal(distance_matrix, 0)

    index = 0
    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            if j in (i + 1, i + 2, i - 1, i - 2):
                distance_matrix[i, j] = 0
                distance_matrix[j, i] = 0
                continue

            distance_matrix[i, j] = distance_vector[index]
            distance_matrix[j, i] = distance_vector[index]
            index += 1

    return distance_matrix


# =====================================================================
# Scaling and test-set preprocessing
# =====================================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Recreate distance matrices for test set
X_test_scaled = np.array(
    [recreate_distance_matrix_exclude_neighbors(row) for row in X_test_scaled]
)
X_test_scaled = X_test_scaled.reshape(
    X_test_scaled.shape[0],
    X_test_scaled.shape[1],
    X_test_scaled.shape[2],
    1,
)


# =====================================================================
# K-fold cross-validation setup
# =====================================================================

kf = KFold(n_splits=5, shuffle=True)

fold = 1
custom_learning_rate = 1e-4

all_train_accuracy, all_train_f1, all_train_loss = [], [], []
all_test_accuracy, all_test_f1, all_test_loss = [], [], []

train_losses_per_fold = []
val_losses_per_fold = []

analysis_list = []
prediction = []


# =====================================================================
# Training loop
# =====================================================================

for train_index, val_index in kf.split(X_train):
    print(f"Fold: {fold}")

    # Split data
    X_train_fold = X_train.iloc[train_index]
    X_val_fold = X_train.iloc[val_index]
    y_train_fold = y_train.iloc[train_index]
    y_val_fold = y_train.iloc[val_index]

    # Scale
    X_train_scaled = scaler.fit_transform(X_train_fold.values)
    X_val_scaled = scaler.transform(X_val_fold.values)

    # Reconstruct matrices
    X_train_scaled = np.array(
        [recreate_distance_matrix_exclude_neighbors(row) for row in X_train_scaled]
    )
    X_val_scaled = np.array(
        [recreate_distance_matrix_exclude_neighbors(row) for row in X_val_scaled]
    )

    X_train_scaled = X_train_scaled[..., np.newaxis]
    X_val_scaled = X_val_scaled[..., np.newaxis]

    # Model definition
    input_shape = X_train_scaled.shape[1:]
    input_layer = Input(shape=input_shape)

    x = layers.Conv2D(
        2, (2, 2), activation="relu",
        kernel_regularizer=regularizers.l2(0.1)
    )(input_layer)
    x = Dropout(0.4)(x)

    x = layers.Conv2D(
        4, (2, 2), activation="relu",
        kernel_regularizer=regularizers.l2(0.1)
    )(x)
    x = Dropout(0.4)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(4, activation="relu",
                     kernel_regularizer=regularizers.l2(0.1))(x)
    x = Dropout(0.4)(x)

    x = layers.Dense(8, activation="relu",
                     kernel_regularizer=regularizers.l2(0.1))(x)
    x = Dropout(0.4)(x)

    logits = layers.Dense(1)(x)
    output = tf.keras.activations.sigmoid(logits)

    model = Model(input_layer, output)

    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=custom_learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train_scaled,
        y_train_fold,
        validation_data=(X_val_scaled, y_val_fold),
        epochs=2000,
        batch_size=256,
        verbose=1,
    )

    # Save model
    model.save(f"model_{fold}.h5")

    # LRP analysis
    logits_model = Model(input_layer, logits)
    lrp = innvestigate.create_analyzer(
        "lrp.sequential_preset_a", logits_model
    )

    analysis = np.abs(lrp.analyze(X_test_scaled))
    analysis_list.append(analysis)

    analysis_flat = analysis.reshape(analysis.shape[0], -1)
    pd.DataFrame(analysis_flat).to_csv(
        f"Relevance values_{fold}.csv", index=False
    )

    # Evaluation
    train_pred = (model.predict(X_train_scaled) > 0.5).astype(int)
    val_pred = (model.predict(X_val_scaled) > 0.5).astype(int)

    train_accuracy = accuracy_score(y_train_fold, train_pred)
    train_f1 = f1_score(y_train_fold, train_pred)
    val_accuracy = accuracy_score(y_val_fold, val_pred)
    val_f1 = f1_score(y_val_fold, val_pred)

    train_loss = history.history["loss"][-1]
    val_loss = history.history["val_loss"][-1]

    print(
        f"Train Acc: {train_accuracy}, Train F1: {train_f1}, Train Loss: {train_loss}"
    )
    print(
        f"CV Acc: {val_accuracy}, CV F1: {val_f1}, CV Loss: {val_loss}"
    )

    all_train_accuracy.append(train_accuracy)
    all_train_f1.append(train_f1)
    all_train_loss.append(train_loss)

    all_test_accuracy.append(val_accuracy)
    all_test_f1.append(val_f1)
    all_test_loss.append(val_loss)

    train_losses_per_fold.append(history.history["loss"])
    val_losses_per_fold.append(history.history["val_loss"])

    fold += 1


# =====================================================================
# Post-processing: loss curves
# =====================================================================

train_losses = np.array(train_losses_per_fold)
val_losses = np.array(val_losses_per_fold)

train_mean = np.mean(train_losses, axis=0)
train_std = np.std(train_losses, axis=0)
val_mean = np.mean(val_losses, axis=0)
val_std = np.std(val_losses, axis=0)

epochs = range(1, train_losses.shape[1] + 1)

plt.figure(figsize=(20, 12))
plt.plot(epochs, np.log(train_mean), label="log(Train)")
plt.fill_between(
    epochs,
    np.log(train_mean - train_std),
    np.log(train_mean + train_std),
    alpha=0.2,
)

plt.plot(epochs, np.log(val_mean), label="log(Validation)")
plt.fill_between(
    epochs,
    np.log(val_mean - val_std),
    np.log(val_mean + val_std),
    alpha=0.2,
)

plt.legend()
plt.savefig("loss_curve.png", bbox_inches="tight")
plt.show()


# =====================================================================
# Final evaluation on test set
# =====================================================================

test_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
test_accuracy = accuracy_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred)

conf_mat = confusion_matrix(y_test, test_pred)

print(f"Test Accuracy: {test_accuracy}, Test F1: {test_f1}")
print("Confusion Matrix:")
print(conf_mat)


# =====================================================================
# Aggregate relevance heatmap
# =====================================================================

analysis_array = np.array(analysis_list)

average = np.mean(analysis_array, axis=0)
std_dev = np.std(analysis_array, axis=0)

average = average.reshape(-1, 138, 138).mean(axis=0)
std_dev = std_dev.reshape(-1, 138, 138).mean(axis=0)

pd.DataFrame(average).to_csv("heatmap.csv")

fig, axes = plt.subplots(1, 2, figsize=(40, 15))

im1 = axes[0].imshow(average, cmap="viridis", aspect="auto")
fig.colorbar(im1, ax=axes[0])
axes[0].set_title("Average Heatmap")

im2 = axes[1].imshow(std_dev, cmap="plasma", aspect="auto")
fig.colorbar(im2, ax=axes[1])
axes[1].set_title("Std Dev Heatmap")

plt.savefig("heatmap.png", bbox_inches="tight")
plt.show()


# =====================================================================
# Symmetry correlation (upper vs lower triangle)
# =====================================================================

matrix = pd.read_csv("heatmap.csv", header=None).iloc[1:, 1:].values

upper = matrix[np.triu_indices_from(matrix, k=1)]
lower = matrix[np.tril_indices_from(matrix, k=-1)]

corr, pval = spearmanr(upper, lower)

print(f"Spearman correlation: {corr}, p-value: {pval}")
