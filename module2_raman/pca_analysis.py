"""
Principal Component Analysis of Raman spectra.

This corresponds to Exercise 3 of the assignment, which involved:
  1. Selecting a range of Raman shifts (x-values) to analyse
  2. Computing the correlation matrix of the combined data
  3. Eigendecomposing the correlation matrix to get principal components
  4. Plotting PC1 vs PC2 scatter to visually assess class separation
  5. Comparing multiple spectral ranges to find the one with best separation

Why PCA?
--------
Each Raman spectrum has hundreds of wavenumber points, but most of these
are correlated (nearby wavenumbers have similar intensities). PCA finds the
directions of maximum variance in the data — the "principal components" —
which are linear combinations of the original wavenumber variables.

PC1 captures the most variance, PC2 the second most, and so on.
A scatter plot of PC1 vs PC2 shows the two most important axes of variation
in the dataset. If cancer and healthy samples cluster separately in this
2D view, it means the first two PCs capture biochemically meaningful
differences — and that these differences should be exploitable for
classification.

The assignment method: correlation matrix → eigendecomposition
-------------------------------------------------------------
The assignment computed PCA via:
  1. corrcoef() — the correlation matrix (normalised covariance)
  2. eig()      — eigendecomposition to get eigenvectors and eigenvalues

This is mathematically equivalent to scikit-learn's PCA (which uses SVD
of the centred data matrix), but the intermediate steps are more transparent.
We implement BOTH approaches here so you can see they give the same result.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAResult:
    """
    Output of a single PCA run on a given spectral range.

    scores:       (N, n_components) — the data projected onto the PCs.
                  Each row is one spectrum; each column is one PC score.
                  This is what you plot as PC1 vs PC2.
    eigenvalues:  (n_wavenumbers,) — sorted descending. The proportion of
                  variance explained by PCk is eigenvalues[k] / sum(eigenvalues).
    eigenvectors: (n_wavenumbers, n_wavenumbers) — the loading vectors.
                  Column k is the kth principal component direction in the
                  original wavenumber space (the "spectral fingerprint" of PCk).
    labels:       (N,) — class labels (1=cancer, 0=healthy) matching rows of scores.
    spectral_range: (wn_min, wn_max) tuple identifying which range was used.
    variance_explained: fraction of total variance captured by each PC.
    """
    scores: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    labels: np.ndarray
    spectral_range: tuple[float, float]
    variance_explained: np.ndarray


def run_pca_correlation(
    cancer: np.ndarray,
    healthy: np.ndarray,
    raman_shifts: np.ndarray,
    wn_min: float | None = None,
    wn_max: float | None = None,
    n_components: int = 10,
) -> PCAResult:
    """
    PCA via correlation matrix eigendecomposition — matches the assignment method.

    Steps:
      1. Select spectral range (rows in MATLAB = wavenumber axis here)
      2. Concatenate cancer and healthy spectra: [Healthy; Cancer] (MATLAB order)
      3. Standardise: the assignment noted data was "already standardised" —
         here we z-score each wavenumber channel to unit variance
      4. Compute correlation matrix: corrcoef() in MATLAB ≡ np.corrcoef() in Python
      5. Eigendecompose: eig() in MATLAB ≡ np.linalg.eigh() in Python
         (eigh is for symmetric matrices — correlation matrices are always symmetric)
      6. Sort eigenvalues descending (MATLAB's eig() sorts ascending — we flip)
      7. Project data onto top eigenvectors to get PC scores

    Parameters
    ----------
    wn_min, wn_max:
        Spectral range in cm⁻¹. None = use full range.
        In the assignment you tested 1:600, 1:800, 800:end, and full range.
        Here you specify actual wavenumber values (more interpretable).
    """
    wn_min = wn_min if wn_min is not None else float(raman_shifts[0])
    wn_max = wn_max if wn_max is not None else float(raman_shifts[-1])

    # Select the spectral range
    mask = (raman_shifts >= wn_min) & (raman_shifts <= wn_max)
    cancer_sub  = cancer[:, mask]
    healthy_sub = healthy[:, mask]

    # Concatenate — assignment did [Healthy(range,:); Cancer(range,:)]
    data = np.vstack([healthy_sub, cancer_sub])   # (N, n_wavenumbers_in_range)
    labels = np.array([0] * len(healthy) + [1] * len(cancer))

    # Remove constant columns (zero variance) before standardising.
    # Real instruments often record zeros at the spectrum edges where the
    # detector has no sensitivity. A column of all-zeros has std=0, causing
    # divide-by-zero in both StandardScaler and np.corrcoef, producing NaN
    # values that corrupt the entire eigendecomposition.
    col_std = data.std(axis=0)
    active  = col_std > 1e-10
    data    = data[:, active]

    # Standardise each wavenumber channel (column) to zero mean, unit variance.
    # This is important because wavenumber channels have different mean intensities.
    # Without standardisation, high-intensity peaks would dominate the correlation
    # structure regardless of how discriminative they are.
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)   # (N, n_active_wavenumbers)

    # Correlation matrix: (n_wavenumbers × n_wavenumbers)
    # Each entry [i,j] is the Pearson correlation between wavenumber i and j
    # across all N spectra. It measures how often two wavenumber channels
    # vary together — i.e. whether they carry redundant information.
    # Note: np.corrcoef expects (variables × observations), so we transpose.
    corr_matrix = np.corrcoef(data_std.T)    # matches MATLAB: corrcoef(stand_data)

    # Eigendecomposition of the symmetric correlation matrix.
    # eigh() is faster and more numerically stable than eig() for symmetric matrices.
    # Returns eigenvalues in ASCENDING order — we reverse to get descending.
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    eigenvalues  = eigenvalues[::-1]          # largest first
    eigenvectors = eigenvectors[:, ::-1]      # columns reordered to match

    # Keep only positive eigenvalues (small negatives can appear due to float errors)
    eigenvalues = np.maximum(eigenvalues, 0)

    # Variance explained by each PC
    total_var = eigenvalues.sum()
    var_explained = eigenvalues / (total_var + 1e-12)

    # Project data onto the top n_components principal components.
    # Each score is a dot product of the standardised spectrum with an eigenvector.
    # This converts the high-dimensional spectrum into a low-dimensional coordinate.
    scores = data_std @ eigenvectors[:, :n_components]   # (N, n_components)

    return PCAResult(
        scores=scores,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors[:, :n_components],
        labels=labels,
        spectral_range=(wn_min, wn_max),
        variance_explained=var_explained[:n_components],
    )


def compute_class_separation(pca_result: PCAResult) -> float:
    """
    Quantify how well PC1 and PC2 separate the two classes.

    Uses the Fisher discriminant ratio on the 2D PC space:
        FDR = (μ_cancer - μ_healthy)² / (σ_cancer² + σ_healthy²)

    A higher FDR means better class separation — this is the objective
    version of "visually inspecting the PC1 vs PC2 scatter plot" that
    the assignment asked you to do by eye. Used to rank spectral ranges
    automatically in optimise_spectral_range().
    """
    c_mask = pca_result.labels == 1
    h_mask = pca_result.labels == 0

    c_scores = pca_result.scores[c_mask, :2]
    h_scores = pca_result.scores[h_mask, :2]

    mu_diff = c_scores.mean(axis=0) - h_scores.mean(axis=0)
    sigma   = c_scores.var(axis=0)  + h_scores.var(axis=0) + 1e-12

    return float(np.sum(mu_diff ** 2 / sigma))


def optimise_spectral_range(
    cancer: np.ndarray,
    healthy: np.ndarray,
    raman_shifts: np.ndarray,
    ranges: list[tuple[float | None, float | None]] | None = None,
) -> list[PCAResult]:
    """
    Run PCA over multiple spectral ranges and rank by class separation.

    This implements the core iterative analysis from Exercise 3:
    the assignment asked you to test the full range, 1:600, 1:800, and 800:end.
    Here the ranges are specified in cm⁻¹ rather than array indices.

    Default ranges chosen to mimic the assignment's test cases, translated
    from index space to the actual wavenumber axis of our data.

    Returns a list of PCAResult objects, sorted by class separation (best first).
    """
    if ranges is None:
        full_min = float(raman_shifts[0])
        full_max = float(raman_shifts[-1])
        midpoint = float(np.median(raman_shifts))
        ranges = [
            (full_min, full_max),            # full range (assignment: 1:end)
            (full_min, midpoint),            # lower half (assignment: 1:~600)
            (full_min, midpoint * 0.85),     # shorter lower range (assignment: 1:800 approx)
            (midpoint, full_max),            # upper half (assignment: 800:end)
        ]

    results = []
    for wn_min, wn_max in ranges:
        r = run_pca_correlation(cancer, healthy, raman_shifts, wn_min, wn_max)
        results.append(r)

    results.sort(key=compute_class_separation, reverse=True)
    return results


def plot_pc_scores(
    pca_result: PCAResult,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """
    PC1 vs PC2 scatter plot with class colour-coding.

    This is the key diagnostic plot from Exercise 3. If the two clouds of
    points (cancer = red, healthy = blue) are well separated, the spectral
    range captures biochemically meaningful differences that a classifier
    can exploit. Overlapping clouds mean the range contains too much noise
    or uninformative spectral regions.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    c_mask = pca_result.labels == 1
    h_mask = pca_result.labels == 0

    ax.scatter(
        pca_result.scores[h_mask, 0], pca_result.scores[h_mask, 1],
        c='steelblue', alpha=0.6, s=30, label='Healthy', edgecolors='none',
    )
    ax.scatter(
        pca_result.scores[c_mask, 0], pca_result.scores[c_mask, 1],
        c='salmon', alpha=0.6, s=30, label='Cancer', edgecolors='none',
    )

    wn_min, wn_max = pca_result.spectral_range
    ax.set_xlabel(f'PC1 ({pca_result.variance_explained[0]*100:.1f}% var)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca_result.variance_explained[1]*100:.1f}% var)', fontsize=11)
    ax.set_title(title or f'PC1 vs PC2  [{wn_min:.0f}–{wn_max:.0f} cm⁻¹]', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    sep = compute_class_separation(pca_result)
    ax.text(0.02, 0.96, f'Separation score: {sep:.2f}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return fig


def plot_pca_comparison(
    results: list[PCAResult],
    n_cols: int = 2,
) -> plt.Figure:
    """
    Grid of PC1 vs PC2 scatter plots for multiple spectral ranges.

    Replicates the side-by-side comparison the assignment asked for in Q3.1.
    The best-separating range is highlighted with a green border.
    """
    n = len(results)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    axes = np.array(axes).flatten()

    best_idx = 0  # results are pre-sorted by separation (best first)

    for i, (ax, result) in enumerate(zip(axes, results)):
        plot_pc_scores(result, ax=ax)
        if i == best_idx:
            for spine in ax.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)
            ax.set_title(ax.get_title() + '  ✓ best', color='green', fontsize=12)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle('PCA Range Optimisation — PC1 vs PC2', fontsize=14, y=1.01)
    plt.tight_layout()
    return fig


def plot_eigenvalues(
    pca_result: PCAResult,
    n_components: int = 10,
) -> plt.Figure:
    """
    Scree plot: eigenvalue magnitude vs PC index.

    The "elbow" in this plot indicates how many PCs are needed to capture
    the meaningful variance in the data. PCs beyond the elbow are dominated
    by noise. In the assignment this was one of the two required plots for
    Exercise 3; it guides how many PC scores to use as features for
    classification in Exercise 4.
    """
    n = min(n_components, len(pca_result.eigenvalues))
    pc_indices = np.arange(1, n + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Scree plot (raw eigenvalues)
    ax1.plot(pc_indices, pca_result.eigenvalues[:n], 'bo-', markersize=7)
    ax1.axvline(2.5, color='grey', linestyle=':', alpha=0.5)  # mark first 2 PCs
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('Scree plot — eigenvalues')
    ax1.set_xticks(pc_indices)
    ax1.grid(alpha=0.3)

    # Cumulative variance explained
    cumvar = np.cumsum(pca_result.variance_explained[:n]) * 100
    ax2.bar(pc_indices, pca_result.variance_explained[:n] * 100,
            alpha=0.6, color='steelblue', label='Per-PC')
    ax2.step(pc_indices, cumvar, where='mid', color='red',
             linewidth=2, label='Cumulative')
    ax2.axhline(90, color='grey', linestyle='--', alpha=0.6, label='90% threshold')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Variance explained (%)')
    ax2.set_title('Variance explained')
    ax2.set_xticks(pc_indices)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig
