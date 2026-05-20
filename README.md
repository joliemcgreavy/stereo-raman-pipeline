# stereo-raman-pipeline — Stereo 3D Reconstruction & Raman Tissue Classification

A Python pipeline demonstrating two core perception tasks for next-generation surgical robots:

1. **Stereo 3D reconstruction** — calibrate a stereo endoscopic camera pair, estimate depth from disparity, and localise a target tissue region in 3D space
2. **Raman spectral tissue classification** — load cancer and healthy cell spectra, extract discriminative features via PCA, and compare supervised ML classifiers

The two modules mirror a real clinical workflow: the robot's stereo camera identifies *where* a suspicious tissue region is, and a co-located Raman spectroscopy probe characterises *what* it is.

---

## Background

This project translates concepts from a Clinical Engineering and Surgical Robotics module at Imperial College London into an open-source, reproducible Python pipeline. The original coursework used MATLAB GUI tools (Stereo Camera Calibrator, Classification Learner); this project reimplements every step programmatically using industry-standard Python libraries.

---

## Pipeline Overview

```
Stereo images (SERV-CT / Hamlyn)          Raman spectra (MDA-MB-231 cells)
        │                                          │
        ▼                                          ▼
 Module 1: Stereo Vision                  Module 2: Spectral Analysis
 ├── Stereo calibration (OpenCV)          ├── Mean ± std visualisation
 ├── Intrinsic/extrinsic parameters       ├── Manual peak analysis
 ├── Single-image pose estimation         ├── PCA (range optimisation)
 ├── Disparity map                        └── SVM / KNN / RF classifiers
 └── Q-matrix → 3D target coordinates         (Acc, Precision, Sensitivity,
                                               Specificity, F-score, ROC/AUC)
        │                                          │
        └──────────────┬───────────────────────────┘
                       ▼
              Module 3: Streamlit Dashboard
              (3D scene + classification results)
```

---

## Datasets

| Module | Dataset | Source |
|--------|---------|--------|
| Stereo calibration | SERV-CT | UCL / arXiv 2012.11779 |
| Stereo 3D scene | Hamlyn Centre Endoscopic Dataset | Imperial College London |
| Raman spectroscopy | MDA-MB-231 breast cancer cells | RamanSPy / Imperial Barahona Group |

See `data/README.md` for download instructions. Raw data is not committed to this repository.

---

## Project Structure

```
surgical-vision-pipeline/
├── module1_stereo/          # Stereo vision Python modules
│   ├── calibration.py
│   ├── pose_estimation.py
│   ├── disparity.py
│   └── reconstruction.py
├── module2_raman/           # Raman spectroscopy Python modules
│   ├── loader.py
│   ├── peak_analysis.py
│   ├── pca_analysis.py
│   └── classification.py
├── module3_dashboard/       # Streamlit app
│   └── app.py
├── notebooks/               # Step-by-step annotated notebooks
│   ├── 01_stereo_calibration.ipynb
│   ├── 02_3d_reconstruction.ipynb
│   ├── 03_raman_analysis.ipynb
│   └── 04_dashboard_integration.ipynb
├── tests/                   # Unit tests
├── data/                    # Data directory (raw data not committed)
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/surgical-vision-pipeline
cd surgical-vision-pipeline

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Then follow `data/README.md` to download the datasets.

---

## Usage

Run notebooks interactively:
```bash
jupyter notebook notebooks/
```

Run the Streamlit dashboard:
```bash
streamlit run module3_dashboard/app.py
```

Run tests:
```bash
pytest tests/
```

---

## Tech Stack

Python 3.10+ · OpenCV · NumPy · SciPy · scikit-learn · RamanSPy · Open3D · Matplotlib · Plotly · Streamlit

---

## Acknowledgements

- [RamanSPy](https://ramanspy.readthedocs.io) — Barahona Research Group, Imperial College London
- [Hamlyn Centre Dataset](http://hamlyn.doc.ic.ac.uk/vision/) — Hamlyn Centre, Imperial College London
- [SERV-CT](https://arxiv.org/abs/2012.11779) — Psychogyios et al., UCL
