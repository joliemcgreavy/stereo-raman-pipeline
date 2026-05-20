"""
Raman spectroscopy data loading and preprocessing.

The assignment loaded two fixed .txt files (AT1_data_norm.txt for cancer,
Fibro_data_norm.txt for healthy) that had already been normalised by the
course team. Here we do the same thing but with four options:

  1. COVID-19 serum Raman (primary real dataset) — Yin et al. (2021),
     Journal of Raman Spectroscopy. 159 COVID-positive and 150 healthy
     serum spectra, 400–2112 cm⁻¹. Directly analogous to the cancer vs
     healthy classification task in the assignment. Downloaded from Figshare
     (public, no login required). See data/README.md for instructions.

  2. RamanSPy — loads datasets via the Imperial College London Barahona
     Research Group's open-source library. Note: all RamanSPy datasets
     require manual pre-download; see ramanspy.readthedocs.io/datasets.

  2. Synthetic (fallback / demo) — generates realistic cancer vs healthy
     spectra from published tissue Raman peak positions. Useful for running
     the pipeline without downloading anything, and for understanding the
     spectral features that drive classification.

  3. Custom files — loads two .txt files in the same format as the assignment
     data, so you can plug in any real dataset later.

Why normalise?
--------------
Raw Raman intensities vary with laser power, integration time, sample
concentration, and path length. Normalising removes these experimental
variables so that only the *shape* of the spectrum (the relative peak
intensities) matters for classification. The assignment data was
pre-normalised; here we apply vector normalisation (divide each spectrum
by its L2 norm) which is standard practice.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RamanDataset:
    """
    Container for a two-class Raman spectroscopy dataset.

    cancer:        (N_cancer, n_wavenumbers) normalised spectra
    healthy:       (N_healthy, n_wavenumbers) normalised spectra
    raman_shifts:  (n_wavenumbers,) wavenumber axis in cm⁻¹
    source:        string describing the dataset origin (for plot labels)
    """
    cancer: np.ndarray
    healthy: np.ndarray
    raman_shifts: np.ndarray
    source: str

    @property
    def n_wavenumbers(self) -> int:
        return len(self.raman_shifts)

    @property
    def n_cancer(self) -> int:
        return len(self.cancer)

    @property
    def n_healthy(self) -> int:
        return len(self.healthy)

    def combined(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (X, y) ready for scikit-learn.

        Stacks both classes into one matrix, assigns labels:
          1 = cancer,  0 = healthy
        This is the same structure as the assignment's classification table,
        where each row was an observation and the last column was the class label.
        """
        X = np.vstack([self.cancer, self.healthy])
        y = np.array([1] * self.n_cancer + [0] * self.n_healthy)
        return X, y


# ── Option 1: COVID-19 serum Raman (Yin et al. 2021) ──────────────────────

def load_covid_raman(
    data_dir: Path | str | None = None,
) -> RamanDataset:
    """
    Load the Yin et al. (2021) COVID-19 serum Raman dataset.

    Published in: Journal of Raman Spectroscopy, 52(5), 949–958.
    Source: https://figshare.com/articles/dataset/12159924

    Two classes:
      - COVID-positive serum  → used as the "disease" class (159 spectra)
      - Healthy control serum → used as the "healthy" class (150 spectra)

    This is directly analogous to the cancer vs healthy classification in
    the assignment. The technique (serum Raman spectroscopy) and the
    analytical task (binary classification from spectral features) are
    identical; the biological substrate is serum rather than tissue.

    Spectral range: 400–2112 cm⁻¹, 900 wavenumber points.

    Download instructions (data/README.md):
      The three required files are already downloaded if you ran the
      setup script. They live in data/raw/raman_covid/:
        raw_COVID.txt      — 159 COVID spectra  (900 wavenumbers × 159 cols)
        raw_Healthy.txt    — 150 healthy spectra (900 wavenumbers × 150 cols)
        Raman_shift.txt    — 900 wavenumber values

    Parameters
    ----------
    data_dir:
        Path to the folder containing the three .txt files.
        Defaults to data/raw/raman_covid/ relative to the project root.
    """
    if data_dir is None:
        # Navigate from this file's location up to project root, then to data
        data_dir = Path(__file__).parent.parent / 'data' / 'raw' / 'raman_covid'
    data_dir = Path(data_dir)

    required = ['raw_COVID.txt', 'raw_Healthy.txt', 'Raman_shift.txt']
    missing  = [f for f in required if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files in {data_dir}: {missing}\n"
            f"See data/README.md for download instructions."
        )

    # Each file is (wavenumbers × spectra) — same layout as assignment .txt files.
    # Transpose to (spectra × wavenumbers) which is scikit-learn's convention.
    covid_raw   = np.loadtxt(data_dir / 'raw_COVID.txt').T    # (159, 900)
    healthy_raw = np.loadtxt(data_dir / 'raw_Healthy.txt').T  # (150, 900)
    shifts      = np.loadtxt(data_dir / 'Raman_shift.txt').ravel()

    # The spectra are already baseline-corrected and normalised by the
    # original authors. We apply L2 normalisation on top to ensure unit
    # norm, which makes the data consistent with load_synthetic() output.
    return RamanDataset(
        cancer=normalise_spectra(covid_raw),
        healthy=normalise_spectra(healthy_raw),
        raman_shifts=shifts,
        source="Yin et al. (2021) COVID-19 serum Raman — Figshare/JRS",
    )


# ── Option 2: RamanSPy ─────────────────────────────────────────────────────

def load_ramanspy() -> RamanDataset:
    """
    Load the MDA-MB-231 breast cancer cell dataset via RamanSPy.

    MDA-MB-231 is a triple-negative breast adenocarcinoma cell line —
    directly comparable to the AT1 adenocarcinoma cells used in the assignment.

    The dataset is a Raman *image* (each pixel has a full spectrum), so we
    treat each pixel as an independent observation. To create two classes
    we use the spatial structure: nuclear regions (high DNA/RNA content,
    characteristic of cancer cell nuclei) vs cytoplasmic regions (higher
    lipid content, more "healthy-tissue-like" biochemistry).

    This is a simplification — in a real study you would have separate
    cancer and healthy cell lines — but it demonstrates the full pipeline
    and uses real measured spectra.
    """
    try:
        import ramanspy
    except ImportError:
        raise ImportError(
            "RamanSPy is not installed. Run: pip install ramanspy\n"
            "Or use load_synthetic() for a demo dataset."
        )

    train_data = ramanspy.datasets.MDA_MB_231_cells(dataset='train')

    # train_data is a SpectralImage — access the underlying array and axis.
    # Shape: (height, width, n_wavenumbers)
    spectra_cube = train_data.spectral_data          # 3D array
    raman_shifts = train_data.spectral_axis          # 1D wavenumber array

    # Flatten spatial dimensions: (H, W, C) → (H*W, C)
    n_h, n_w, n_c = spectra_cube.shape
    all_spectra = spectra_cube.reshape(-1, n_c)

    # Split into two pseudo-classes using intensity at the DNA/RNA peak
    # (~785 cm⁻¹ is a nucleic acid marker — higher in cell nuclei).
    # Pixels above the median are "nuclear" (cancer-like); below are "cytoplasm".
    # This is a simplified labelling strategy for demonstration purposes.
    dna_peak_idx = np.argmin(np.abs(raman_shifts - 785))
    dna_intensity = all_spectra[:, dna_peak_idx]
    threshold = np.median(dna_intensity)

    cancer_mask  = dna_intensity >= threshold
    healthy_mask = dna_intensity <  threshold

    cancer  = normalise_spectra(all_spectra[cancer_mask])
    healthy = normalise_spectra(all_spectra[healthy_mask])

    return RamanDataset(
        cancer=cancer,
        healthy=healthy,
        raman_shifts=raman_shifts,
        source="RamanSPy / MDA-MB-231 (Kallepitis et al. 2017)",
    )


# ── Option 2: Synthetic ────────────────────────────────────────────────────

def load_synthetic(
    n_cancer: int = 120,
    n_healthy: int = 120,
    noise_level: float = 0.02,
    rng_seed: int = 42,
) -> RamanDataset:
    """
    Generate synthetic cancer and healthy tissue Raman spectra.

    Peak positions are taken from published tissue Raman spectroscopy
    literature (primarily Movasaghi et al. 2007, Applied Spectroscopy Reviews).
    Relative intensities reflect typical cancer vs healthy tissue differences:

      Cancer cells:
        - Higher nucleic acid peaks (785, 1090 cm⁻¹) — DNA/RNA replication
        - Higher protein peaks (1002, 1268, 1655 cm⁻¹) — elevated metabolism
        - Lower lipid peaks (1302, 1444, 1745 cm⁻¹) — membrane changes

      Healthy/fibroblast cells:
        - Higher lipid peaks — intact membrane structure
        - Lower nucleic acid content
        - Higher collagen peaks (852, 937 cm⁻¹) — stromal tissue

    This mirrors the AT1 (adenocarcinoma) vs Fibro (fibroblast) contrast
    from the assignment, which showed the same biochemical pattern.

    Parameters
    ----------
    n_cancer, n_healthy:
        Number of spectra per class. 120 each gives a meaningful dataset
        that scikit-learn classifiers can generalise from.
    noise_level:
        Standard deviation of Gaussian noise added to each spectrum.
        Real Raman spectra have shot noise (photon counting statistics)
        plus detector read noise.
    """
    rng = np.random.default_rng(rng_seed)

    # Wavenumber axis: fingerprint region 600–1800 cm⁻¹, sampled every 2 cm⁻¹
    # This is the most diagnostically relevant range for tissue
    raman_shifts = np.arange(600, 1802, 2, dtype=np.float64)
    n_wn = len(raman_shifts)

    def gaussian(centre: float, width: float, amplitude: float) -> np.ndarray:
        """Single Lorentzian-shaped peak (approximated as Gaussian for simplicity)."""
        return amplitude * np.exp(-((raman_shifts - centre) ** 2) / (2 * width ** 2))

    # ── Shared background (both tissue types) ─────────────────────────────
    background = (
        gaussian(852,  15, 0.12) +   # proline/hydroxyproline (collagen)
        gaussian(937,  14, 0.10) +   # C-C backbone stretch (proteins)
        gaussian(1002, 12, 0.35) +   # phenylalanine ring breathing
        gaussian(1127, 15, 0.15) +   # C-N stretch (proteins)
        gaussian(1444, 16, 0.45) +   # CH₂ bending (lipids + proteins)
        gaussian(1655, 22, 0.50)     # amide I (C=O stretch, proteins)
    )

    # ── Cancer-specific spectral signature ────────────────────────────────
    cancer_signature = background + (
        gaussian(785,  14, 0.30) +   # nucleic acids (DNA/RNA ring breathing)
        gaussian(1090, 15, 0.22) +   # C-O-C stretch (nucleic acids, phosphate)
        gaussian(1268, 18, 0.28) +   # amide III (higher in cancer — more protein)
        gaussian(1302, 14, 0.15) +   # CH₂ twist (lipids — slightly lower)
        gaussian(1745, 14, 0.08)     # C=O ester (lipids — much lower in cancer)
    )

    # ── Healthy-specific spectral signature ───────────────────────────────
    healthy_signature = background + (
        gaussian(785,  14, 0.10) +   # nucleic acids (much lower in healthy)
        gaussian(1090, 15, 0.08) +
        gaussian(1268, 18, 0.18) +   # amide III (lower — less protein turnover)
        gaussian(1302, 14, 0.30) +   # CH₂ twist (lipids — higher in healthy)
        gaussian(1445, 16, 0.10) +   # extra lipid CH₂ contribution
        gaussian(1745, 14, 0.28)     # C=O ester (lipids — much higher in healthy)
    )

    def make_spectra(template: np.ndarray, n: int) -> np.ndarray:
        """
        Create n spectra by adding per-spectrum random variation to a template.

        The variation has two parts:
          - Amplitude scaling (~5% variation): simulates different cell sizes
            or laser focus positions
          - Additive Gaussian noise: simulates shot noise
        We then L2-normalise each spectrum, as was done in the assignment.
        """
        scales = rng.normal(1.0, 0.05, size=(n, 1))           # per-spectrum scaling
        noise  = rng.normal(0.0, noise_level, size=(n, n_wn))  # shot noise
        spectra = scales * template[None, :] + noise
        spectra = np.clip(spectra, 0, None)                    # intensities are non-negative
        return normalise_spectra(spectra)

    cancer  = make_spectra(cancer_signature,  n_cancer)
    healthy = make_spectra(healthy_signature, n_healthy)

    return RamanDataset(
        cancer=cancer,
        healthy=healthy,
        raman_shifts=raman_shifts,
        source="Synthetic (peaks from Movasaghi et al. 2007)",
    )


# ── Option 3: Custom files ─────────────────────────────────────────────────

def load_from_files(
    cancer_path: Path,
    healthy_path: Path,
    shifts_path: Path,
) -> RamanDataset:
    """
    Load Raman data from three .txt files — same format as the assignment.

    Each spectrum occupies one column; rows are wavenumber points.
    The assignment files (AT1_data_norm.txt, Fibro_data_norm.txt) used this
    layout. str2num() in MATLAB reads this as a matrix; np.loadtxt() does
    the same in Python.

    The returned spectra are transposed so each ROW is one spectrum,
    which is the convention scikit-learn expects (samples × features).
    """
    cancer  = np.loadtxt(cancer_path).T
    healthy = np.loadtxt(healthy_path).T
    shifts  = np.loadtxt(shifts_path).ravel()
    return RamanDataset(
        cancer=normalise_spectra(cancer),
        healthy=normalise_spectra(healthy),
        raman_shifts=shifts,
        source=f"Custom ({cancer_path.name} / {healthy_path.name})",
    )


# ── Shared utilities ───────────────────────────────────────────────────────

def normalise_spectra(spectra: np.ndarray) -> np.ndarray:
    """
    L2-normalise each spectrum (divide each row by its Euclidean norm).

    After normalisation every spectrum has unit length in the
    n_wavenumber-dimensional space. This removes intensity differences
    caused by concentration or laser power, leaving only spectral *shape*
    information — the part that carries chemical identity.

    The assignment data was pre-normalised; this function applies the
    same operation to any raw data.
    """
    norms = np.linalg.norm(spectra, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # avoid divide-by-zero
    return spectra / norms


def spectral_range_indices(
    raman_shifts: np.ndarray,
    wn_min: float,
    wn_max: float,
) -> np.ndarray:
    """
    Return the indices of raman_shifts that fall within [wn_min, wn_max].

    This is the Python equivalent of MATLAB's range indexing used in
    Exercise 3: 'range = value1:value2' selected rows of the data matrix.
    Here we select by actual wavenumber values rather than arbitrary indices,
    which is more physically meaningful.
    """
    return np.where((raman_shifts >= wn_min) & (raman_shifts <= wn_max))[0]
