"""
Manual peak analysis of Raman spectra.

This corresponds to Exercises 1 and 2 of the assignment:

  Exercise 1: Compute and plot average ± standard deviation curves
              for the Cancer and Healthy classes.

  Exercise 2: Manually select 4 key spectral peaks that differ between
              classes, extract intensities at those peaks, and compute
              Cancer:Healthy intensity ratios.

The word "manual" is important — you are using domain knowledge to pick
peaks that are biochemically meaningful, not letting an algorithm choose
for you. This is exactly what a spectroscopist does when interpreting data.

In the assignment you did this by eye, looking at the plotted average
spectra. Here we provide the same visual output plus an automated
find_discriminative_peaks() helper that ranks candidate peaks by how
different they are between classes — useful when you have many peaks
and aren't sure which four to pick.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import find_peaks


def compute_mean_std(spectra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the mean and standard deviation spectrum for a class.

    Each column of the input is one wavenumber; each row is one spectrum.
    np.mean(axis=0) averages across all spectra to give the mean intensity
    at each wavenumber — this is exactly what MATLAB's mean(Cancer) did
    when Cancer was organised as (wavenumbers × observations).

    The standard deviation captures inter-sample variability: how much
    individual spectra deviate from the average shape. A narrow std band
    means the class is spectrally consistent; a wide band means high
    biological variability.
    """
    return spectra.mean(axis=0), spectra.std(axis=0)


def plot_mean_spectra(
    cancer: np.ndarray,
    healthy: np.ndarray,
    raman_shifts: np.ndarray,
    peak_shifts: list[float] | None = None,
    title: str = "Mean Raman spectra: Cancer vs Healthy",
) -> plt.Figure:
    """
    Plot mean ± std spectra for both classes.

    Replicates the stdshade() visualisation from the assignment:
    a solid mean line with a semi-transparent shaded band representing ±1 std.

    Parameters
    ----------
    peak_shifts:
        Optional list of wavenumber values to highlight as vertical lines.
        Pass your four selected peaks here after running Exercise 2.
    """
    cancer_avg,  cancer_std  = compute_mean_std(cancer)
    healthy_avg, healthy_std = compute_mean_std(healthy)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Healthy: blue (solid mean line + shaded ±1 std band)
    ax.plot(raman_shifts, healthy_avg, 'b-', linewidth=1.5, label='Mean Healthy')
    ax.fill_between(
        raman_shifts,
        healthy_avg - healthy_std,
        healthy_avg + healthy_std,
        alpha=0.25, color='blue', label='±1 STD Healthy',
    )

    # Cancer: red (dashed mean line + shaded ±1 std band)
    ax.plot(raman_shifts, cancer_avg, 'r--', linewidth=1.5, label='Mean Cancer')
    ax.fill_between(
        raman_shifts,
        cancer_avg - cancer_std,
        cancer_avg + cancer_std,
        alpha=0.25, color='red', label='±1 STD Cancer',
    )

    # Mark selected peaks
    if peak_shifts:
        for i, wn in enumerate(peak_shifts):
            ax.axvline(wn, color='green', linestyle=':', alpha=0.8, linewidth=1.2)
            ax.text(wn + 5, ax.get_ylim()[1] * 0.85,
                    f'P{i+1}\n{wn:.0f}', fontsize=8, color='darkgreen')

    ax.set_xlabel('Raman shift (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Normalised intensity', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(raman_shifts[0], raman_shifts[-1])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def find_discriminative_peaks(
    cancer: np.ndarray,
    healthy: np.ndarray,
    raman_shifts: np.ndarray,
    n_peaks: int = 4,
    min_peak_height: float = 0.05,
    min_peak_distance_cm: float = 50.0,
) -> list[float]:
    """
    Algorithmically rank candidate peaks by Cancer:Healthy ratio.

    This does not replace visual inspection — it assists it. The steps are:
      1. Find all local maxima in the mean cancer spectrum
      2. Compute the Cancer:Healthy intensity ratio at each maximum
      3. Return the top n_peaks by ratio magnitude (furthest from 1.0)

    "Furthest from 1.0" selects peaks that are either much higher in cancer
    (ratio >> 1) or much lower in cancer (ratio << 1). Both types are
    discriminative for classification.

    Parameters
    ----------
    min_peak_height:
        Ignore peaks below this normalised intensity (avoids noise peaks).
    min_peak_distance_cm:
        Minimum separation between peaks in cm⁻¹ (avoids selecting the
        same peak multiple times due to noise). 50 cm⁻¹ is a reasonable
        minimum for distinct Raman bands.
    """
    cancer_avg,  _ = compute_mean_std(cancer)
    healthy_avg, _ = compute_mean_std(healthy)

    # Convert cm⁻¹ distance to index distance
    wn_step = float(np.mean(np.diff(raman_shifts)))
    min_distance_idx = max(1, int(min_peak_distance_cm / wn_step))

    peak_indices, _ = find_peaks(
        cancer_avg,
        height=min_peak_height,
        distance=min_distance_idx,
    )

    # Compute ratio at each peak, avoiding divide-by-zero
    eps = 1e-9
    ratios = cancer_avg[peak_indices] / (healthy_avg[peak_indices] + eps)

    # Sort by how far the ratio is from 1.0
    discriminability = np.abs(np.log(ratios))   # log ratio: 0 = no difference
    sorted_idx = np.argsort(discriminability)[::-1]
    top_indices = peak_indices[sorted_idx[:n_peaks]]

    return [float(raman_shifts[i]) for i in sorted(top_indices)]


def extract_peak_intensities(
    spectra: np.ndarray,
    raman_shifts: np.ndarray,
    peak_shifts: list[float],
) -> np.ndarray:
    """
    Extract the mean spectrum intensity at each selected peak wavenumber.

    Finds the closest wavenumber in raman_shifts to each requested peak_shift
    value. This is the Python equivalent of:
        P1 = find(Raman_shift == X1)   % MATLAB index lookup
    but more robust — it finds the nearest value rather than requiring an
    exact match, which avoids the errors the assignment code warned about.

    Returns an array of shape (n_peaks,) containing mean intensities.
    """
    avg, _ = compute_mean_std(spectra)
    intensities = []
    for wn in peak_shifts:
        idx = int(np.argmin(np.abs(raman_shifts - wn)))
        intensities.append(float(avg[idx]))
    return np.array(intensities)


def compute_cancer_healthy_ratios(
    cancer: np.ndarray,
    healthy: np.ndarray,
    raman_shifts: np.ndarray,
    peak_shifts: list[float],
) -> dict:
    """
    Compute Cancer:Healthy intensity ratios at selected peaks.

    This directly implements what the assignment called ratio_P1 through
    ratio_P4: the ratio of mean cancer intensity to mean healthy intensity
    at each manually-selected peak wavenumber.

    A ratio > 1 means the peak is elevated in cancer.
    A ratio < 1 means the peak is suppressed in cancer (elevated in healthy).
    Both are diagnostically useful — the assignment highlighted that the
    lipid peaks (elevated in healthy/fibroblasts) are just as informative
    as the nucleic acid peaks (elevated in cancer).

    Returns
    -------
    Dictionary with keys: 'wavenumbers', 'cancer_intensities',
    'healthy_intensities', 'ratios', 'best_peak_idx'
    """
    c_int = extract_peak_intensities(cancer,  raman_shifts, peak_shifts)
    h_int = extract_peak_intensities(healthy, raman_shifts, peak_shifts)
    ratios = c_int / (h_int + 1e-9)

    return {
        'wavenumbers':         peak_shifts,
        'cancer_intensities':  c_int,
        'healthy_intensities': h_int,
        'ratios':              ratios,
        'best_peak_idx':       int(np.argmax(np.abs(np.log(ratios)))),
    }


def plot_peak_analysis(
    result: dict,
    cancer: np.ndarray,
    healthy: np.ndarray,
    raman_shifts: np.ndarray,
) -> plt.Figure:
    """
    Two-panel figure showing per-peak intensities and Cancer:Healthy ratios.

    Left panel: bar chart of cancer vs healthy mean intensity at each peak.
    Right panel: bar chart of Cancer:Healthy ratio (dashed line at ratio=1).

    The dashed line at 1.0 is important — it marks "no difference". Bars
    above 1.0 indicate cancer-elevated peaks; bars below 1.0 indicate
    healthy-elevated peaks.
    """
    wns    = result['wavenumbers']
    c_int  = result['cancer_intensities']
    h_int  = result['healthy_intensities']
    ratios = result['ratios']
    labels = [f'P{i+1}\n({wn:.0f} cm⁻¹)' for i, wn in enumerate(wns)]
    x      = np.arange(len(wns))
    width  = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: intensities
    bars_c = ax1.bar(x - width/2, c_int, width, label='Cancer',  color='salmon',      alpha=0.85)
    bars_h = ax1.bar(x + width/2, h_int, width, label='Healthy', color='steelblue',   alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel('Mean normalised intensity')
    ax1.set_title('Peak intensities: Cancer vs Healthy')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Right: ratios
    colors = ['salmon' if r >= 1 else 'steelblue' for r in ratios]
    ax2.bar(x, ratios, color=colors, alpha=0.85, edgecolor='k', linewidth=0.5)
    ax2.axhline(1.0, color='black', linestyle='--', linewidth=1.2, label='Ratio = 1 (no difference)')
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel('Cancer : Healthy intensity ratio')
    ax2.set_title('Cancer:Healthy intensity ratios')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # Annotate best peak
    best = result['best_peak_idx']
    ax2.patches[best].set_edgecolor('gold')
    ax2.patches[best].set_linewidth(2.5)
    ax2.text(best, ratios[best] + 0.02, '★ best', ha='center', fontsize=9, color='darkgreen')

    cancer_patch  = mpatches.Patch(color='salmon',    label='Elevated in Cancer')
    healthy_patch = mpatches.Patch(color='steelblue', label='Elevated in Healthy')
    ax2.legend(handles=[cancer_patch, healthy_patch], fontsize=9)

    plt.tight_layout()
    return fig


def print_peak_summary(result: dict) -> None:
    """Print a table matching the format of Assignment Q2.1–2.3."""
    print("=" * 58)
    print(f"{'Peak':<6} {'Shift (cm⁻¹)':<16} {'Cancer':<12} {'Healthy':<12} {'Ratio':<8}")
    print("-" * 58)
    for i, (wn, c, h, r) in enumerate(zip(
        result['wavenumbers'],
        result['cancer_intensities'],
        result['healthy_intensities'],
        result['ratios'],
    )):
        marker = " ← highest" if i == result['best_peak_idx'] else ""
        print(f"P{i+1:<5} {wn:<16.1f} {c:<12.4f} {h:<12.4f} {r:<8.4f}{marker}")
    print("=" * 58)
    best_wn = result['wavenumbers'][result['best_peak_idx']]
    print(f"\nHighest Cancer:Healthy ratio at {best_wn:.1f} cm⁻¹ "
          f"(ratio = {result['ratios'][result['best_peak_idx']]:.4f})")
