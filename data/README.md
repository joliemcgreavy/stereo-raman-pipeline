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

### COVID-19 Serum Raman Dataset (primary — already downloaded)
- **Paper**: Yin G. et al., *Journal of Raman Spectroscopy*, 52(5), 949–958 (2021)
- **Source**: https://figshare.com/articles/dataset/12159924
- **Licence**: CC BY 4.0 (open access)
- **Files**: `data/raw/raman_covid/raw_COVID.txt`, `raw_Healthy.txt`, `Raman_shift.txt`
- **Download** (if files are missing):
  ```bash
  mkdir -p data/raw/raman_covid && cd data/raw/raman_covid
  curl -L -o raw_COVID.txt   https://ndownloader.figshare.com/files/22386432
  curl -L -o raw_Healthy.txt https://ndownloader.figshare.com/files/22386435
  curl -L -o data.mat        https://ndownloader.figshare.com/files/22386411
  python3 -c "import scipy.io,numpy; d=scipy.io.loadmat('data.mat'); numpy.savetxt('Raman_shift.txt', d['wave_number'].ravel())"
  ```
- **Note**: Two classes — COVID-positive (disease, n=159) and healthy controls (n=150).
  Directly analogous to the cancer vs healthy classification task.
  Spectral range 400–2112 cm⁻¹, 900 wavenumber points.
