# ProtXAI: Explainable AI Analysis of Protein Dynamics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21872520.svg)](https://doi.org/10.5281/zenodo.21872520)

* **Manuscript:** [ProtXAI: Explainable AI Reveals Structural Determinants of Protein Dynamics](https://doi.org/10.64898/2026.05.26.727866)
* **Zenodo repository:** https://zenodo.org/records/21872520
* **Loschmidt Laboratories:** https://loschmidt.chemi.muni.cz/

This repository contains the code and implementation for three case studies using the **ProtXAI pipeline** for the analysis of molecular dynamics (MD) data. The framework combines machine learning and explainable AI (XAI) to identify structural determinants of protein dynamics.

The three case studies are:

* **ApoE4** – CNN-based classification using residue–residue distance-difference features.
* **SAK** – CNN-based classification of four SAK variants using distance-difference features.
* **Luciferase** – next-snapshot prediction using a convolutional autoencoder and Cα coordinates.

## Requirements

The analyses are implemented in Python and use libraries including:

* NumPy
* pandas
* SciPy
* scikit-learn
* TensorFlow / Keras
* PyEMMA
* MDTraj
* Matplotlib
* BioPython
* **iNNvestigate**

Some preprocessing steps additionally require **cpptraj**.

Please refer to the README files in the individual case-study directories for more details.

## Case studies

### ApoE4

The ApoE4 case study uses a CNN to distinguish ApoE4 from ApoE4 + 3-SPA based on residue–residue distance-difference features. Layer-wise Relevance Propagation (LRP) is used to identify important residue interactions.

**Main file:** [`Clean code_ApoE4.py`](./ApoE4/Clean%20code_ApoE4.py)

For detailed information, see the [`ApoE4/README.md`](./ApoE4/README.md).

### SAK

The SAK case study analyzes four SAK variants using CNN classification based on residue–residue distance-difference features. LRP is used to identify relevant structural interactions.

**Main file:** [`SAK_clean code.py`](./SAK/SAK_clean%20code.py)

For detailed information, see the [`SAK/README.md`](./SAK/README.md).

### Luciferase

The Luciferase case study uses a convolutional autoencoder to predict the next molecular-dynamics snapshot from the current snapshot using Cα coordinates. LRP is used to identify residues associated with the learned protein dynamics.

**Main file:** [`Clean_anc.py`](./LUC/Clean_anc.py)

For detailed information, see the [`LUC/README.md`](./LUC/README.md).

## Contact

For questions or further information, please contact:

**Faraneh Haddadi**
Loschmidt Laboratories, Masaryk University
Email: [faranehhaddadi@gmail.com](mailto:faranehhaddadi@gmail.com)

