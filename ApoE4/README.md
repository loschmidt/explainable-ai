# ApoE4 -- Adaptive Sampling CNN Analysis with XAI

##  Systems

Two systems were analyzed:

-   ApoE4 (protein only)
-   ApoE4 + 3-SPA (protein--ligand)

Adaptive sampling trajectories were used.

------------------------------------------------------------------------

##  Input

### MD simulations

-   Topology: `filtered.pdb`
-   Trajectories: multiple adaptive sampling folders (`e*s*`)

### Train/test split (trajectory level)

The bash preprocessing:

-   splits folders → **80% train / 20% test**
-   ignores the first 10% of frames in the test set
-   prevents data leakage

------------------------------------------------------------------------

##  Workflow

### 1. Trajectory preprocessing (cpptraj + bash)

For each trajectory:

-   strip unwanted atoms
-   align and center the protein
-   subsample frames

Output:

-   stripped and aligned `.short.xtc` trajectories
-   separate **train** and **test** directories

------------------------------------------------------------------------

### 2. Feature extraction (PyEMMA)

For each trajectory:

-   compute residue--residue minimum distances
-   compute **difference between consecutive frames**

This captures **protein dynamics**.

Output:

CSV file containing:

    frames × distance-difference features

------------------------------------------------------------------------

### 3. Dataset construction

-   Merge all trajectories for each system
-   Add labels:

```{=html}
<!-- -->
```
    0 → ApoE4
    1 → ApoE4 + 3-SPA

Final files:

-   `difference_of_distance_train.csv`
-   `difference_of_distance_test.csv`

------------------------------------------------------------------------

### 4. CNN training

-   Binary classification
-   5-fold cross-validation (train set)
-   Independent test-set evaluation
-   Input: reconstructed distance matrices

------------------------------------------------------------------------

### 5. Explainable AI (XAI)

Layer-wise Relevance Propagation (**LRP**) is used to:

-   identify important residue--residue interactions
-   generate relevance heatmaps

------------------------------------------------------------------------

##  Output

-   `model_<fold>.h5` → trained CNN models
-   `Relevance values_<fold>.csv` → LRP scores
-   `loss_curve.png`
-   `heatmap.csv`
-   `heatmap.png`
-   Test accuracy, F1-score, confusion matrix
-   Spearman correlation of relevance matrix symmetry

------------------------------------------------------------------------

##  Requirements

Python with:

-   numpy
-   pandas
-   scikit-learn
-   tensorflow / keras
-   pyemma
-   innvestigate
-   scipy
-   matplotlib
-   seaborn

------------------------------------------------------------------------

##  Notes

-   Train/test split is done **before feature generation**
-   Prevents trajectory-level data leakage
-   The model learns **frame-to-frame structural changes**, not static
    structures
