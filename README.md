---
title: Stereo Raman Pipeline
emoji: 🔬
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# stereo-raman-pipeline

**Stereo 3D reconstruction and Raman spectroscopy tissue classification for surgical robotics**

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HF%20Spaces-blue)](https://huggingface.co/spaces/joliemcgreavy/stereo-raman-pipeline)
[![Tests](https://img.shields.io/badge/tests-43%20passing-brightgreen)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

An end-to-end Python pipeline demonstrating two core perception tasks for next-generation surgical robots:

1. **Stereo 3D reconstruction** — load a rectified stereo endoscope image pair, compute a dense disparity map using SGBM, project to 3D via the Q matrix, and localise a target tissue region in millimetre coordinates. Validated against CT ground truth.

2. **Raman spectral tissue classification** — load real serum Raman spectra, apply baseline correction and cosmic ray removal, extract PCA features, train and compare six ML classifiers, and output a prediction with confidence score.

The two modules reflect a real clinical workflow: the stereo camera identifies *where* a suspicious region is; the Raman probe characterises *what* it is.

---

## Live Demo

**[→ Open the interactive dashboard](https://huggingface.co/spaces/joliemcgreavy/stereo-raman-pipeline)**

The dashboard runs all modules live, including:
- SERV-CT stereo frame selector (16 real endoscope image pairs)
- SGBM disparity map vs CT ground-truth validation
- Interactive 3D point cloud with selectable tissue target
- Raman classification on real published data
- Preprocessing, learning curves, and uncertainty quantification tabs

---

## Results

| Module | Metric | Value |
|--------|--------|-------|
| Stereo (Module 1) | Disparity MAE vs CT | ~3–5 px |
| Stereo (Module 1) | Depth MAE vs CT | ~5 mm |
| Raman (Module 2) | Best classifier accuracy | **94.8%** (Random Forest) |
| Raman (Module 2) | AUC | **0.980** |
| Raman (Module 2) | Sensitivity | 0.912 |
| Raman (Module 2) | Specificity | 0.987 |
| Test suite | | 43 / 43 passing |

---

## Datasets

| Module | Dataset | Source | Licence |
|--------|---------|--------|---------|
| Stereo | SERV-CT — 16 rectified stereo pairs, ex vivo porcine tissue, da Vinci™ endoscope, CT ground-truth depth | [UCL / Psychogyios et al. 2022](https://rdr.ucl.ac.uk/articles/dataset/26352199) | CC BY 4.0 |
| Raman | Yin et al. 2021 — 309 serum Raman spectra (159 disease, 150 healthy), 400–2112 cm⁻¹ | [Figshare / J. Raman Spectrosc.](https://figshare.com/articles/dataset/12159924) | CC BY 4.0 |

Both datasets are downloaded automatically on first run. See `data/README.md` for manual download instructions.

---

## Pipeline

```
Stereo image pair (SERV-CT)           Raman spectra (Yin et al. 2021)
        │                                       │
        ▼                                       ▼
Module 1: Stereo Vision               Module 2: Spectral Analysis
├── Calibration (Q, P1, P2)           ├── Baseline correction (ALS)
├── SGBM disparity map                ├── Cosmic ray removal
├── CT ground-truth validation        ├── Mean ± std visualisation
├── Q-matrix → 3D coordinates         ├── Manual peak analysis
└── Interactive 3D point cloud        ├── PCA range optimisation
                                      ├── SVM / KNN / RF / NB / DT / LR
                                      ├── Confusion matrix, ROC, AUC
                                      ├── Learning curves
                                      └── Uncertainty quantification
        │                                       │
        └─────────────┬─────────────────────────┘
                      ▼
           Module 3: Streamlit Dashboard
           Interactive stereo + classification results
```

---

## Project Structure

```
stereo-raman-pipeline/
├── module1_stereo/
│   ├── calibration.py        stereo calibration from checkerboard images
│   ├── pose_estimation.py    single-image probe distance estimation (solvePnP)
│   ├── disparity.py          SGBM dense disparity + manual single-point
│   ├── reconstruction.py     Q-matrix reprojection to 3D
│   ├── serv_ct_loader.py     SERV-CT dataset loader (images, calibration, GT)
│   └── validation.py         MAE, RMSE, >1px vs CT ground truth
├── module2_raman/
│   ├── loader.py             COVID-19 Raman loader + synthetic fallback
│   ├── preprocessing.py      ALS baseline correction, cosmic ray removal
│   ├── peak_analysis.py      mean spectra, peak selection, C:H ratios
│   ├── pca_analysis.py       correlation matrix PCA, range optimisation
│   └── classification.py     6 classifiers, all metrics, learning curves, uncertainty
├── module3_dashboard/
│   └── app.py                Streamlit dashboard (7 Module 2 tabs, real stereo data)
├── notebooks/
│   ├── 01_stereo_calibration.ipynb
│   ├── 02_stereo_real_data.ipynb
│   ├── 03_raman_analysis.ipynb
│   ├── 03b_raman_extensions.ipynb
│   └── 04_dashboard_integration.ipynb
├── data/
│   ├── downloader.py         auto-downloads datasets on first run
│   └── README.md             manual download instructions
└── tests/
    ├── test_calibration.py   14 tests — disparity, Q-matrix, point cloud
    └── test_classification.py 29 tests — loader, PCA, metrics, classifiers
```

---

## Setup

```bash
git clone https://github.com/joliemcgreavy/stereo-raman-pipeline
cd stereo-raman-pipeline

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**Run the dashboard:**
```bash
streamlit run module3_dashboard/app.py
```
Datasets download automatically on first launch (~40MB total).

**Run the notebooks:**
```bash
pip install jupyter
jupyter notebook notebooks/
```

**Run the tests:**
```bash
pytest tests/ -v
```

---

## Tech Stack

Python 3.11 · OpenCV · NumPy · SciPy · scikit-learn · Matplotlib · Plotly · Streamlit

---

## Background

Developed as part of graduate study in Clinical Engineering and Surgical Robotics at Imperial College London. Translates concepts from stereo camera calibration and Raman spectral classification into an open-source Python pipeline using published open datasets from the UCL and Imperial research communities.
