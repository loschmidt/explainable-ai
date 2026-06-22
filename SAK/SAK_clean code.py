# -*- coding: utf-8 -*-
"""
Multiclass CNN classification of residue–residue distance differences
with K-fold cross-validation and LRP analysis.

CLEANED VERSION
---------------
- ALL original logic preserved
- NO steps removed
- NO behavior changed
- ONLY formatting, comments, and deduplication
"""

# ======================================================================
# Imports
# ======================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical

from scipy.stats import spearmanr
import innvestigate


# ======================================================================
# Load data
# ======================================================================

df = pd.read_csv("difference_merged_output.csv", header=None)

X = df.iloc[:, :-1].values   # Features
y = df.iloc[:, -1].values   # Labels


# ======================================================================
# Encode labels (4 classes)
# ======================================================================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded, num_classes=4)


# ======================================================================
# Train / test split and scaling
# ======================================================================

scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, shuffle=True
)

X_train_scaled1 = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ======================================================================
# Distance matrix reconstruction
# ======================================================================

num_residues = 136

def recreate_distance_matrix_exclude_neighbors(distance_vector,
                                               num_residues=num_residues):
    """
    Reconstruct symmetric residue–residue distance matrix
    excluding diagonal and |i-j| in {1, 2}.
    """
    distance_matrix = np.full((num_residues, num_residues), np.inf)
    np.fill_diagonal(distance_matrix, 0)

    index = 0
    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            if abs(j - i) in [1, 2]:
                distance_matrix[i, j] = 0
                distance_matrix[j, i] = 0
                continue

            distance_matrix[i, j] = distance_vector[index]
            distance_matrix[j, i] = distance_vector[index]
            index += 1

    return distance_matrix


X_train_processed = np.array(
    [recreate_distance_matrix_exclude_neighbors(row)
     for row in X_train_scaled1]
)

X_test_processed = np.array(
    [recreate_distance_matrix_exclude_neighbors(row)
     for row in X_test_scaled]
)

X_train_scaled1 = X_train_processed[..., np.newaxis]
X_test_scaled = X_test_processed[..., np.newaxis]


# ======================================================================
# Sort test set by labels and save
# ======================================================================

y_test_labels = np.argmax(y_test, axis=1)

sorted_indices = np.argsort(y_test_labels)
X_test_sorted = X_test_scaled[sorted_indices]
y_test_sorted = y_test_labels[sorted_indices]

test_data = np.hstack(
    (X_test_sorted.reshape(len(X_test_sorted), -1),
     y_test_sorted.reshape(-1, 1))
)

np.savetxt(
    "test_set_sorted_distance.csv",
    test_data,
    delimiter=",",
    fmt="%.6f"
)

print("Test set saved as 'test_set_sorted_distance.csv'")


# ======================================================================
# K-fold cross-validation setup
# ======================================================================

kf = KFold(n_splits=5, shuffle=True)

custom_learning_rate = 1e-5
fold = 1

all_train_accuracy, all_train_f1, all_train_loss = [], [], []
all_test_accuracy, all_test_f1, all_test_loss = [], [], []

train_losses_per_fold = []
val_losses_per_fold = []

analysis_list = []


# ======================================================================
# Training loop
# ======================================================================

for train_index, val_index in kf.split(X_train):

    print(f"Fold: {fold}")

    X_train_fold = X_train[train_index]
    X_val_fold = X_train[val_index]
    y_train_fold = y_train[train_index]
    y_val_fold = y_train[val_index]

    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_val_scaled = scaler.transform(X_val_fold)

    X_train_scaled = np.array(
        [recreate_distance_matrix_exclude_neighbors(row)
         for row in X_train_scaled]
    )
    X_val_scaled = np.array(
        [recreate_distance_matrix_exclude_neighbors(row)
         for row in X_val_scaled]
    )

    X_train_scaled = X_train_scaled[..., np.newaxis]
    X_val_scaled = X_val_scaled[..., np.newaxis]

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    input_layer = Input(shape=X_train_scaled.shape[1:])

    x = layers.Conv2D(
        2, (2, 2), activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.1)
    )(input_layer)
    x = Dropout(0.05)(x)

    x = layers.Conv2D(
        4, (2, 2), activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.1)
    )(x)
    x = Dropout(0.05)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(
        8, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.1)
    )(x)
    x = Dropout(0.05)(x)

    logits_layer = layers.Dense(4)(x)
    output_layer = tf.keras.activations.softmax(logits_layer)

    model = Model(input_layer, output_layer)

    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=custom_learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train_scaled,
        y_train_fold,
        validation_data=(X_val_scaled, y_val_fold),
        epochs=3000,
        batch_size=8,
        verbose=1
    )

    # ------------------------------------------------------------------
    # LRP analysis
    # ------------------------------------------------------------------

    logits_model = Model(input_layer, logits_layer)
    model.save(f"model_distance_{fold}.h5")

    lrp = innvestigate.create_analyzer(
        "lrp.sequential_preset_a", logits_model
    )

    analysis1 = np.abs(lrp.analyze(X_test_sorted))
    analysis_list.append(analysis1)

    analysis1 = analysis1.reshape(
        analysis1.shape[0],
        analysis1.shape[1] * analysis1.shape[2]
    )

    pd.DataFrame(analysis1).to_csv(
        f"Relevance values_distance_{fold}.csv"
    )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]

    train_loss = history.history["loss"][-1]
    val_loss = history.history["val_loss"][-1]

    y_train_pred = np.argmax(model.predict(X_train_scaled), axis=1)
    y_val_pred = np.argmax(model.predict(X_val_scaled), axis=1)

    y_train_true = np.argmax(y_train_fold, axis=1)
    y_val_true = np.argmax(y_val_fold, axis=1)

    f1_train = f1_score(y_train_true, y_train_pred, average="weighted")
    f1_val = f1_score(y_val_true, y_val_pred, average="weighted")

    print(f"Fold {fold} Results:")
    print(f"Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")
    print(f"Training Loss: {train_loss:.9f}, Validation Loss: {val_loss:.9f}")
    print(f"F1 Score (Train): {f1_train:.4f}, F1 Score (Validation): {f1_val:.4f}")

    all_train_accuracy.append(train_acc)
    all_train_f1.append(f1_train)
    all_train_loss.append(train_loss)
    all_test_accuracy.append(val_acc)
    all_test_f1.append(f1_val)
    all_test_loss.append(val_loss)

    train_losses_per_fold.append(history.history["loss"])
    val_losses_per_fold.append(history.history["val_loss"])

    fold += 1


# ======================================================================
# Save metrics and losses
# ======================================================================

pd.DataFrame({
    "train_accuracy": all_train_accuracy,
    "train_f1": all_train_f1,
    "train_loss": all_train_loss,
    "test_accuracy": all_test_accuracy,
    "test_f1": all_test_f1,
    "test_loss": all_test_loss
}).to_csv("metrics_distance.csv", index=False)

pd.DataFrame({
    "train_losses_per_fold": train_losses_per_fold,
    "val_losses_per_fold": val_losses_per_fold
}).to_csv("losses_distance.csv", index=False)


# ======================================================================
# Aggregate LRP heatmap
# ======================================================================

analysis_array = np.array(analysis_list)

average = np.mean(analysis_array, axis=0)
std_dev = np.std(analysis_array, axis=0)

average = average.reshape(average.shape[0], 136, 136)
std_dev = std_dev.reshape(std_dev.shape[0], 136, 136)

average = np.mean(average, axis=0)
std_dev = np.mean(std_dev, axis=0)

pd.DataFrame(average).to_csv("heatmap_distance.csv")

fig, axes = plt.subplots(1, 2, figsize=(40, 15))

im1 = axes[0].imshow(average, cmap="viridis", aspect="auto")
fig.colorbar(im1, ax=axes[0])
axes[0].set_title("Average Heatmap")

im2 = axes[1].imshow(std_dev, cmap="plasma", aspect="auto")
fig.colorbar(im2, ax=axes[1])
axes[1].set_title("Standard Deviation Heatmap")

plt.savefig("heatmap_distance.png", bbox_inches="tight")
plt.show()


# ======================================================================
# Spearman correlation (EXPLICITLY INCLUDED)
# ======================================================================

matrix = pd.read_csv("heatmap_distance.csv", header=None).iloc[1:, 1:].values

upper_triangle = []
lower_triangle = []

for i in range(matrix.shape[0]):
    for j in range(i + 1, matrix.shape[1]):
        upper_triangle.append(matrix[i, j])

for j in range(matrix.shape[1]):
    for i in range(j + 1, matrix.shape[0]):
        lower_triangle.append(matrix[i, j])

upper_triangle = np.array(upper_triangle)
lower_triangle = np.array(lower_triangle)

correlation, p_value = spearmanr(upper_triangle, lower_triangle)

print(f"Spearman's correlation: {correlation}")
print(f"P-value: {p_value}")

with open("distance_distance.txt", "a") as f:
    f.write(f"correlation: {correlation}\n")
    f.write(f"p-value: {p_value}\n")
