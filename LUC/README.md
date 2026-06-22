# Luciferase -- Next-Snapshot Prediction with Autoencoder + XAI

## 🧬 System

RLuc8 luciferase molecular dynamics trajectories.

The trajectories are **pre-aligned and solvent-free** before dataset
construction.

This pipeline predicts **snapshot t+1 from snapshot t** using Cα
coordinates.

⚠️ No PyEMMA is used in this workflow.

------------------------------------------------------------------------

## 📥 Input

### MD simulations

-   Topology: `filtered.pdb`
-   Multiple aligned trajectories from adaptive sampling

### Experimental / reference data

-   Experimental B-factors
-   Calculated B-factors
-   Reference PDB structure

------------------------------------------------------------------------

## ⚙️ Workflow

### 1. Dataset construction (MDTraj)

For each trajectory:

-   Extract Cα atoms
-   Sample snapshots with a fixed gap
-   Create input--label pairs:

Input → snapshot t\
Label → snapshot t+1

Final dataset:

`dataset.csv`

------------------------------------------------------------------------

### 2. Train / test split

-   Random shuffle
-   80% training
-   20% testing

------------------------------------------------------------------------

### 3. Autoencoder training

Architecture:

-   1D convolutional encoder
-   Bottleneck representation
-   1D transposed-convolution decoder

Task:

Predict **next structural snapshot**.

Training:

-   10 independent runs
-   MSE loss
-   Adam optimizer

------------------------------------------------------------------------

### 4. Explainable AI (XAI)

Layer-wise Relevance Propagation (**LRP**) applied to:

-   full autoencoder output
-   encoder bottleneck

Used to identify residues driving structural dynamics.

------------------------------------------------------------------------

### 5. Biophysical validation

Relevance profiles are compared with:

-   experimental B-factors
-   calculated B-factors

Using:

-   Spearman correlation
-   secondary-structure--resolved analysis

------------------------------------------------------------------------

## 📤 Output

-   `autoencoder_model_iteration_<i>.h5`
-   `encoder_model_iteration_<i>.h5`
-   `Relevance values_<i>.csv`
-   `Autoencoder_results.xlsx`
-   `encoder_results.xlsx`
-   Loss curves
-   Relevance vs B-factor plots
-   Secondary-structure correlation analysis

------------------------------------------------------------------------

## 🧩 Requirements

Python with:

-   numpy
-   pandas
-   tensorflow / keras
-   mdtraj
-   innvestigate
-   scipy
-   matplotlib
-   scikit-learn
-   BioPython

------------------------------------------------------------------------

## 🧠 Notes

-   The model learns **protein dynamics in coordinate space**
-   Input features: Cα coordinates (885 values)
-   Output: predicted next snapshot coordinates
-   XAI connects learned dynamics with experimental flexibility
