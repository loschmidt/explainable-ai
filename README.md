# ProtXAI: Explainable AI Analysis of Protein Dynamics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21872520.svg)](https://doi.org/10.5281/zenodo.21872520)

* **Manuscript:** [ProtXAI: Explainable AI Reveals Structural Determinants of Protein Dynamics](https://doi.org/10.64898/2026.05.26.727866)
* **Zenodo repository:** https://zenodo.org/records/21872520
* **Loschmidt Laboratories:** https://loschmidt.chemi.muni.cz/

The repository contains the code and implementation for three case studies using the **ProtXAI pipeline** for the analysis of molecular dynamics (MD) data. The framework combines machine learning and explainable AI (XAI) to identify structural determinants of protein dynamics.

## Requirements

The analyses are implemented in Python and use libraries including:

* NumPy
* pandas
* SciPy
* scikit-learn
* TensorFlow / Keras
* PyEMMA
* MDTraj
* BioPython
* Matplotlib
* **iNNvestigate** for explainable AI and Layer-wise Relevance Propagation (LRP)
* Seaborn

Some preprocessing steps additionally require **cpptraj**.

For information about iNNvestigate and installation, see the [iNNvestigate repository](https://github.com/albermax/innvestigate).

## Case studies

### ApoE4

CNN-based classification of ApoE4 and ApoE4 + 3-SPA using residue–residue distance-difference features, followed by LRP analysis.

**Main file:** [`Clean code_ApoE4.py`](./ApoE4/Clean%20code_ApoE4.py)

See [`ApoE4/README.md`](./ApoE4/README.md) for details.

### SAK

CNN-based classification of four SAK variants using residue–residue distance-difference features, followed by LRP analysis.

**Main file:** [`SAK_clean code.py`](./SAK/SAK_clean%20code.py)

See [`SAK/README.md`](./SAK/README.md) for details.

### Luciferase

Convolutional autoencoder for next-snapshot prediction using Cα coordinates, followed by LRP analysis of the learned protein dynamics.

**Main file:** [`Clean_anc.py`](./LUC/Clean_anc.py)

See [`LUC/README.md`](./LUC/README.md) for details.

## Contact

**Faraneh Haddadi**
Loschmidt Laboratories, Masaryk University
Email: [faranehhaddadi@gmail.com](mailto:faranehhaddadi@gmail.com)
