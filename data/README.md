# Data

Raw data files are **not committed to this repository** (see `.gitignore`).
Download each dataset manually using the instructions below and place files
in the directories shown.

---

## Module 1 — Stereo Vision

### SERV-CT (stereo calibration + ground-truth depth)
- **Paper**: Psychogyios et al., *Medical Image Analysis*, 2022
- **Source**: https://arxiv.org/abs/2012.11779
- **Download**: https://www.ucl.ac.uk/interventional-surgical-sciences/serv-ct
- **Place in**: `data/raw/serv_ct/`

### Hamlyn Centre Rectified Stereo Dataset (in vivo surgery)
- **Source**: http://hamlyn.doc.ic.ac.uk/vision/
- **Hugging Face mirror**: https://huggingface.co/datasets/Recasens/HamlynRectifiedDataset
- **Place in**: `data/raw/hamlyn/`

---

## Module 2 — Raman Spectroscopy

### MDA-MB-231 Cancer Cell Dataset
- **Source**: RamanSPy (Imperial College London Barahona Research Group)
- **Paper**: Scandinavian et al., *Analytical Chemistry*, 2024
- **Download**: Automatic via `ramanspy.datasets.MDA_MB_231_cells()` — no manual download needed.
