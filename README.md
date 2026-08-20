# ProtXAI

ProtXAI is an explainable artificial intelligence framework for analyzing protein dynamics and identifying structural determinants of protein function from molecular dynamics (MD) simulations.

The framework combines **molecular dynamics, machine learning, and explainable AI (XAI)** to identify residues and residue–residue interactions that contribute to learned protein behavior.

This repository contains the workflows, scripts, models, and analysis files for three protein systems:

* **ApoE4** – CNN classification using distance-difference features
* **SAK** – CNN classification of four protein variants using distance-difference features
* **Luciferase (RLuc8)** – next-snapshot prediction using a convolutional autoencoder

Detailed information about the data, preprocessing, model architectures, parameters, and analysis can be found in the README files within each protein directory.

## Workflows

### ApoE4

[README](./ApoE4/README.md)

Two systems are analyzed:

* **ApoE4**
* **ApoE4 + 3-SPA**

Adaptive-sampling MD trajectories are processed and converted into residue–residue distance-difference features. A CNN is trained to distinguish the two systems using 5-fold cross-validation and an independent test set.

**Layer-wise Relevance Propagation (LRP)** is then applied to the trained models to identify important residue–residue interactions and generate relevance heatmaps.

The workflow includes:

* MD trajectory preprocessing using `cpptraj`
* Trajectory-level train/test splitting
* Feature extraction using PyEMMA
* CNN classification
* LRP-based XAI analysis
* Relevance heatmap generation
* Model performance and correlation analysis

For the complete workflow and file descriptions, see [`ApoE4/`](./ApoE4/).

---

### SAK

[README](./SAK/README.md)

Four SAK variants are analyzed:

* **SAK42D**
* **STAR**
* **SY155**
* **THR174**

Three independent MD replicas are used for each variant. Residue–residue minimum distances are calculated and differences between consecutive frames are used to capture protein dynamics.

A CNN is trained to classify the four variants, followed by LRP analysis to identify important residue–residue interactions.

The workflow includes:

* MD trajectory preprocessing using `cpptraj`
* Feature extraction using MDTraj and PyEMMA
* Distance-difference calculation
* Dataset construction
* CNN classification with 5-fold cross-validation
* LRP-based XAI analysis
* Relevance heatmap generation

For the complete workflow and file descriptions, see [`SAK/`](./SAK/).

---

### Luciferase

[README](./LUC/README.md)

The Luciferase workflow uses **RLuc8** molecular dynamics trajectories to predict the next structural snapshot from the current snapshot.

The trajectories are pre-aligned and solvent-free. Cα coordinates are used as input features:

```text
Snapshot t → Model → Snapshot t+1
```

A convolutional autoencoder is trained using MSE loss and the Adam optimizer. LRP is applied to the trained model to identify residues contributing to the predicted structural dynamics.

The resulting relevance profiles are compared with:

* Experimental B-factors
* Calculated B-factors
* Secondary-structure information

The workflow includes:

* Dataset construction using MDTraj
* Train/test splitting
* Convolutional autoencoder training
* LRP-based XAI analysis
* B-factor comparison
* Secondary-structure analysis

For the complete workflow and file descriptions, see [`LUC/`](./LUC/).

## Machine Learning and XAI

The three workflows use different machine-learning approaches according to the biological question:

| System    | Task                       | Input representation | Model                     |
| --------- | -------------------------- | -------------------- | ------------------------- |
| **ApoE4** | Binary classification      | Distance differences | CNN                       |
| **SAK**   | Multi-class classification | Distance differences | CNN                       |
| **RLuc8** | Next-snapshot prediction   | Cα coordinates       | Convolutional autoencoder |

For the classification workflows, **LRP** is used to identify important structural interactions.

For Luciferase, LRP is used to identify residues contributing to the prediction of structural dynamics.

## Requirements

The workflows are implemented primarily in Python and use several molecular-dynamics, machine-learning, and data-analysis packages, including:

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

Some workflows additionally require **cpptraj** for trajectory preprocessing.

The exact requirements may differ between systems. See the README in each directory for the corresponding dependencies and instructions.

## Repository Structure

```text
.
├── ApoE4/
│   └── README.md
├── SAK/
│   └── README.md
├── LUC/
│   └── README.md
└── README.md
```

Each protein directory contains the scripts and analysis files required for the corresponding workflow.

For detailed information about **input data, preprocessing, feature generation, model training, XAI analysis, output files, and reproducibility**, please refer to the README and files in the respective protein directory.
