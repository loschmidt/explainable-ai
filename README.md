# ProtXAI: Explainable AI for Protein Dynamics

This repository contains the computational workflows and datasets used in **ProtXAI**, a machine-learning and explainable-AI framework for identifying structural determinants of protein dynamics and function from molecular dynamics (MD) simulations.

The repository contains three complementary case studies:

* **ApoE4** — binary classification of ApoE4 with and without the ligand 3-SPA using distance-difference features and CNNs.
* **SAK** — classification of four staphylokinase variants using distance-difference features and CNNs.
* **Luciferase (RLuc8)** — next-snapshot prediction from Cα coordinates using a convolutional autoencoder.

In all cases, **Explainable AI (XAI)** is used to identify the structural features and residues that contribute most strongly to the learned models.

---

## Repository Structure

```text
.
├── ApoE4/
├── SAK/
├── LUC/
└── README.md
```

Each protein/system has its own directory containing the corresponding preprocessing, feature-extraction, machine-learning, XAI, and analysis workflows.

For complete details, including scripts, input/output files, model architectures, parameters, and analysis procedures, please refer to the **README and files within each individual protein directory**.

---

## General Workflow

The overall workflow consists of the following steps:

```text
MD simulations
      ↓
Trajectory preprocessing
      ↓
Feature extraction
      ↓
Dataset construction
      ↓
Machine learning
      ↓
Explainable AI (LRP)
      ↓
Structural / biophysical interpretation
```

### 1. MD trajectory preprocessing

MD trajectories are prepared using tools such as **cpptraj** and **MDTraj**. Depending on the system, preprocessing includes removal of solvent/ions, alignment, centering, and trajectory subsampling.

For ApoE4, trajectory-level train/test splitting is performed before feature generation to avoid data leakage.

### 2. Feature extraction

Two main representations are used:

* **Residue–residue distance differences** for ApoE4 and SAK.
* **Cα Cartesian coordinates** for Luciferase.

For the distance-based workflows, residue–residue distances are calculated and differences between consecutive frames are used to capture **dynamic structural changes** rather than static structural properties.

The Luciferase workflow instead uses consecutive structural snapshots:

```text
Snapshot t  →  Snapshot t+1
```

### 3. Machine learning

Different ML architectures are used according to the task:

| System | Task                       | Representation       | Model                     |
| ------ | -------------------------- | -------------------- | ------------------------- |
| ApoE4  | Binary classification      | Distance differences | CNN                       |
| SAK    | Multi-class classification | Distance differences | CNN                       |
| RLuc8  | Next-snapshot prediction   | Cα coordinates       | Convolutional autoencoder |

Cross-validation and/or independent test sets are used to evaluate model performance.

### 4. Explainable AI

**Layer-wise Relevance Propagation (LRP)** is applied to the trained models to determine which input features contribute most strongly to their predictions.

For the distance-based models, relevance scores are mapped back to **residue–residue interactions** and visualized as relevance heatmaps.

For Luciferase, relevance profiles are analyzed at the residue level and compared with structural flexibility measures.

### 5. Structural and biophysical interpretation

The XAI results are compared with independent structural or experimental information, including:

* Experimental B-factors
* Calculated B-factors
* Secondary structure
* Residue–residue interactions
* Other structural dynamics measures

Correlation analyses, including **Spearman correlation**, are used to evaluate the relationship between model-derived relevance and independent biophysical measurements.

---

## Case Studies

### 🧬 ApoE4

ApoE4 and ApoE4 + 3-SPA are classified using CNNs trained on residue–residue distance-difference features.

LRP is used to identify important residue interactions and generate relevance heatmaps.

**Main objective:** identify structural interactions associated with the presence of 3-SPA and characterize dynamically important regions.

For the complete workflow, parameters, scripts, and output files, see [`ApoE4/`](./ApoE4/).

---

### 🧬 SAK

Four staphylokinase variants are analyzed:

* SAK42D
* STAR
* SY155
* THR174

Three independent MD replicas are used for each variant. CNNs classify the variants based on distance-difference representations, followed by LRP analysis of important residue–residue interactions.

**Main objective:** identify structural interactions and dynamic regions that distinguish the different SAK variants.

For the complete workflow, parameters, scripts, and output files, see [`SAK/`](./SAK/).

---

### 🧬 Luciferase (RLuc8)

The Luciferase workflow uses aligned, solvent-free MD trajectories and Cα coordinates.

A convolutional autoencoder is trained to predict the next structural snapshot:

```text
Snapshot t  →  Autoencoder  →  Snapshot t+1
```

LRP is applied to both the autoencoder output and its latent representation to identify residues associated with the learned dynamics.

The resulting relevance profiles are compared with experimental and calculated B-factors and analyzed according to secondary structure.

**Main objective:** identify residues associated with the learned protein dynamics and evaluate their relationship with experimentally observed structural flexibility.

For the complete workflow, parameters, scripts, and output files, see [`LUC/`](./LUC/).

---

## Software and Dependencies

The workflows are implemented primarily in **Python** and use a combination of molecular-dynamics, machine-learning, and data-analysis libraries, including:

* NumPy
* pandas
* SciPy
* scikit-learn
* TensorFlow / Keras
* PyEMMA
* MDTraj
* BioPython
* iNNvestigate
* Matplotlib
* Seaborn

Additional command-line tools, such as **cpptraj**, are used for MD trajectory preprocessing.

Exact dependencies may differ between case studies. Please refer to the individual directories for system-specific requirements and scripts.

---

## Reproducibility

The repository is organized so that each case study can be examined and reproduced independently.

Because the three systems use different ML tasks and molecular representations, **the detailed methodology is documented separately within each protein directory**.

For detailed information about:

* MD trajectories and preprocessing
* Feature generation
* Train/test splitting
* Dataset formats
* Model architectures
* Training parameters
* XAI analysis
* Output files
* Structural and biophysical validation

please refer to the corresponding README and scripts in:

* [`ApoE4/`](./ApoE4)
* [`SAK/`](./SAK)
* [`LUC/`](./LUC)

---

## Scientific Overview

ProtXAI combines **molecular dynamics, machine learning, and explainable AI** to move from prediction toward mechanistic interpretation.

Rather than using ML only to distinguish protein states or predict structural changes, the framework uses XAI to determine **which residues and structural interactions drive the learned behavior**.

This provides a connection between:

```text
Molecular dynamics
        ↓
Structural representation
        ↓
Machine-learning model
        ↓
XAI relevance
        ↓
Residue-level interpretation
        ↓
Biophysical validation
```

The three case studies demonstrate this approach across different protein systems and ML tasks, providing complementary perspectives on the relationship between **protein dynamics, structure, and function**.
