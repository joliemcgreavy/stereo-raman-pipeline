"""
Raman spectral preprocessing pipeline.

The COVID-19 dataset we use was already baseline-corrected and normalised
by its authors. But in real clinical use, raw spectra come off the instrument
with two problems that must be fixed before any analysis:

  1. Fluorescence baseline — biological tissue fluoresces under laser
     illumination, producing a broad, slowly-varying background signal
     that sits underneath the sharp Raman peaks. Without removing it,
     the baseline dominates the spectrum and drowns out the Raman features.

  2. Cosmic rays — high-energy particles occasionally strike the CCD
     detector, producing extremely sharp, intense spikes at random
     wavenumbers. A single cosmic ray can corrupt peak detection and
     distort PCA loadings.

This module implements both corrections using methods standard in the
Raman spectroscopy literature, so the pipeline can handle genuinely raw
spectra rather than pre-cleaned data.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve


# ── Baseline correction ────────────────────────────────────────────────────

def asymmetric_least_squares(
    spectrum: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    n_iter: int = 10,
) -> np.ndarray:
    """
    Estimate the fluorescence baseline using Asymmetric Least Squares (ALS).

    Published by Eilers & Boelens (2005). The idea is to fit a smooth
    curve to the spectrum, but weight downward deviations much more
    heavily than upward ones. This forces the fit to hug the *bottom*
    of the spectrum — the baseline — rather than the peaks.

    Why asymmetric? Because Raman peaks always sit ABOVE the baseline,
    never below. If we penalise the fit for being above the data (p is
    small, e.g. 0.01), it will naturally track the floor of the spectrum.

    Parameters
    ----------
    spectrum:
        1D array of raw Raman intensities at each wavenumber.
    lam:
        Smoothness parameter. Higher = smoother baseline. Typical range
        1e4–1e7. Too low: baseline follows the peaks. Too high: baseline
        misses genuine curvature.
    p:
        Asymmetry parameter (0 < p < 0.5). Fraction of the penalty
        assigned to points ABOVE the baseline estimate. Small p (e.g.
        0.01) means the baseline mostly fits from below — appropriate for
        Raman where peaks are always positive deviations.
    n_iter:
        Number of reweighting iterations. Convergence is usually reached
        in 5–10 iterations.

    Returns
    -------
    baseline:
        Estimated fluorescence background, same length as spectrum.
    """
    n = len(spectrum)
    spectrum = np.asarray(spectrum, dtype=np.float64)
    # Second-difference penalty matrix — penalises rapid changes in the baseline
    D = diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n), dtype=np.float64)
    H = lam * D.T @ D

    w = np.ones(n, dtype=np.float64)
    baseline = np.zeros(n, dtype=np.float64)
    for _ in range(n_iter):
        W = diags(w, dtype=np.float64)
        Z = (W + H).tocsr()   # CSR format required by spsolve
        baseline = spsolve(Z, w * spectrum)
        # Reweight: points above the current estimate get weight p,
        # points below get weight (1-p). This pushes the fit downward.
        w = np.where(spectrum > baseline, p, 1 - p)

    return baseline


def correct_baseline(
    spectra: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    n_iter: int = 10,
) -> np.ndarray:
    """
    Apply ALS baseline correction to a batch of spectra.

    Each spectrum is corrected independently. The baseline is subtracted
    and the result is clipped to zero (no negative intensities after
    correction — any residual negatives are noise near the baseline).

    Parameters
    ----------
    spectra:
        (N, n_wavenumbers) array of raw spectra.

    Returns
    -------
    corrected:
        (N, n_wavenumbers) baseline-subtracted, non-negative spectra.
    """
    corrected = np.zeros_like(spectra)
    for i, s in enumerate(spectra):
        baseline = asymmetric_least_squares(s, lam=lam, p=p, n_iter=n_iter)
        corrected[i] = np.clip(s - baseline, 0, None)
    return corrected


# ── Cosmic ray removal ─────────────────────────────────────────────────────

def detect_cosmic_rays(
    spectrum: np.ndarray,
    threshold: float = 5.0,
    window: int = 5,
) -> np.ndarray:
    """
    Detect cosmic ray spikes using a modified Z-score on local differences.

    A cosmic ray produces an extremely sharp spike — typically 1–3 pixels
    wide — that is orders of magnitude above the local spectral level.
    Simple global thresholding fails because the spectrum itself has a wide
    intensity range. Instead, we look at the *local* variation:

      1. Compute a local median smooth (rolling median of width `window`)
      2. Take the residual: spectrum - local_median
      3. Compute the MAD (median absolute deviation) of the residual
      4. Flag points where residual / MAD > threshold as cosmic rays

    The MAD is used instead of std because it is robust to outliers —
    the very spikes we're trying to detect would inflate the std,
    making it a poor noise estimate.

    Parameters
    ----------
    threshold:
        Number of MAD units above which a point is flagged. 5.0 catches
        most cosmic rays while avoiding false positives on real peaks.
    window:
        Width of the local median smoothing window (must be odd).

    Returns
    -------
    mask:
        Boolean array, True where a cosmic ray was detected.
    """
    from scipy.ndimage import median_filter
    smooth = median_filter(spectrum, size=window)
    residual = spectrum - smooth
    mad = np.median(np.abs(residual - np.median(residual)))
    if mad < 1e-10:
        return np.zeros(len(spectrum), dtype=bool)
    modified_z = np.abs(residual) / (1.4826 * mad)
    return modified_z > threshold


def remove_cosmic_rays(
    spectrum: np.ndarray,
    threshold: float = 5.0,
    window: int = 5,
) -> np.ndarray:
    """
    Remove cosmic ray spikes by replacing them with local median values.

    Once spikes are detected, each flagged pixel is replaced with the
    median of its non-flagged neighbours. This is a safe repair because
    the Raman spectrum varies slowly between adjacent wavenumbers (the
    instrument's spectral resolution is ~2–5 cm⁻¹), so the local median
    is a good estimate of what the spectrum should look like at that point.

    Returns the corrected spectrum (same length as input).
    """
    from scipy.ndimage import median_filter
    spike_mask = detect_cosmic_rays(spectrum, threshold=threshold, window=window)
    if not spike_mask.any():
        return spectrum.copy()

    corrected = spectrum.copy()
    smooth = median_filter(spectrum, size=window * 3)
    corrected[spike_mask] = smooth[spike_mask]
    return corrected


def preprocess_spectra(
    spectra: np.ndarray,
    remove_spikes: bool = True,
    correct_fluorescence: bool = True,
    normalise: bool = True,
    spike_threshold: float = 5.0,
    als_lam: float = 1e5,
    als_p: float = 0.01,
) -> np.ndarray:
    """
    Full preprocessing pipeline: spike removal → baseline correction → normalisation.

    The order matters:
      1. Remove cosmic rays FIRST — a spike would distort the ALS baseline fit
      2. Correct baseline — removes broad fluorescence background
      3. Normalise — L2 normalisation for classifier compatibility

    Parameters
    ----------
    spectra:
        (N, n_wavenumbers) raw spectra.
    remove_spikes:
        Whether to apply cosmic ray removal.
    correct_fluorescence:
        Whether to apply ALS baseline correction.
    normalise:
        Whether to L2-normalise each spectrum after correction.

    Returns
    -------
    processed:
        (N, n_wavenumbers) cleaned spectra.
    """
    from module2_raman.loader import normalise_spectra

    processed = spectra.copy()

    if remove_spikes:
        processed = np.array([
            remove_cosmic_rays(s, threshold=spike_threshold)
            for s in processed
        ])

    if correct_fluorescence:
        processed = correct_baseline(processed, lam=als_lam, p=als_p)

    if normalise:
        processed = normalise_spectra(processed)

    return processed


# ── Visualisation ──────────────────────────────────────────────────────────

def plot_preprocessing(
    raw: np.ndarray,
    processed: np.ndarray,
    raman_shifts: np.ndarray,
    spectrum_idx: int = 0,
    show_baseline: bool = True,
    als_lam: float = 1e5,
    als_p: float = 0.01,
) -> plt.Figure:
    """
    Three-panel figure showing a single spectrum before and after preprocessing.

    Panel 1: Raw spectrum with estimated ALS baseline overlaid.
    Panel 2: Preprocessed spectrum (baseline removed, spikes corrected).
    Panel 3: Difference (what was removed) — mostly fluorescence background.

    This is the standard visualisation for validating that preprocessing
    removed the background without distorting the Raman peaks.
    """
    raw_s  = raw[spectrum_idx]
    proc_s = processed[spectrum_idx]
    baseline = asymmetric_least_squares(raw_s, lam=als_lam, p=als_p) if show_baseline else None

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Panel 1: raw + baseline
    axes[0].plot(raman_shifts, raw_s, 'steelblue', lw=1, label='Raw spectrum')
    if baseline is not None:
        axes[0].plot(raman_shifts, baseline, 'r--', lw=1.5, label='ALS baseline')
    axes[0].set_ylabel('Intensity (raw)')
    axes[0].set_title(f'Spectrum {spectrum_idx} — raw with ALS baseline')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Panel 2: processed
    axes[1].plot(raman_shifts, proc_s, 'steelblue', lw=1, label='Processed')
    axes[1].axhline(0, color='grey', lw=0.8, linestyle=':')
    axes[1].set_ylabel('Intensity (corrected)')
    axes[1].set_title('After baseline correction and normalisation')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    # Panel 3: removed background
    if baseline is not None:
        axes[2].fill_between(raman_shifts, baseline, alpha=0.4, color='red',
                              label='Removed fluorescence background')
        axes[2].plot(raman_shifts, baseline, 'r-', lw=1)
    axes[2].set_xlabel('Raman shift (cm⁻¹)')
    axes[2].set_ylabel('Background intensity')
    axes[2].set_title('Removed background (fluorescence)')
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    return fig
