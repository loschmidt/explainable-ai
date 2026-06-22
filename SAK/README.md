# SAK -- Distance-Difference CNN Analysis with XAI

##  Systems

Four SAK mutants were analyzed:

-   SAK42d\
-   STAR\
-   SY155\
-   THR174

For each mutant, **3 independent MD replicas** were used.

------------------------------------------------------------------------

##  Input

### MD simulations

-   Topology: `.prmtop`
-   Trajectory: `.nc`

### After preprocessing

-   `stripped.pdb`
-   `stripped.nc`

### Final ML dataset

-   `difference_merged_output.csv`

Each row: - distance--difference features\
- class label (mutant type)

------------------------------------------------------------------------

##  Workflow

### 1. Trajectory preprocessing (cpptraj)

-   Remove solvent and ions\
-   Align and center the protein

**Output:** - `stripped.pdb` - `stripped.nc`

------------------------------------------------------------------------

### 2. Feature extraction (MDTraj + PyEMMA)

For each replica:

-   Subsample trajectory
-   Compute **residue--residue minimum distances**

**Output:**\
frames × 8911 distance features

------------------------------------------------------------------------

### 3. Distance--difference calculation

-   Difference between **two consecutive frames**
-   Captures structural transitions (protein dynamics)

------------------------------------------------------------------------

### 4. Dataset construction

-   Merge:
    -   all replicas
    -   all mutants

**Final dataset:**\
`difference_merged_output.csv`

------------------------------------------------------------------------

### 5. CNN training

-   5-fold cross-validation\
-   Input: reconstructed 136 × 136 distance matrices

------------------------------------------------------------------------

### 6. Explainable AI (XAI)

Layer-wise Relevance Propagation (**LRP**) is used to:

-   identify important residue--residue interactions\
-   generate relevance heatmaps

------------------------------------------------------------------------

##  Output

-   `model_distance_<fold>.h5` → trained CNN models\
-   `Relevance values_distance_<fold>.csv` → LRP scores\
-   `metrics_distance.csv` → accuracy, F1-score, loss\
-   `heatmap_distance.csv` → average relevance map\
-   `heatmap_distance.png`\
-   Spearman correlation of relevance matrix symmetry

------------------------------------------------------------------------

##  Requirements

Python with:

-   numpy\
-   pandas\
-   scikit-learn\
-   tensorflow / keras\
-   mdtraj\
-   pyemma\
-   innvestigate\
-   scipy\
-   matplotlib

------------------------------------------------------------------------

##  Notes

-   Number of residues: **136**\
-   Distance features per frame: **8911**\
-   The model learns **frame-to-frame structural changes**, not static
    structures.
