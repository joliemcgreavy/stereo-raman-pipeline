"""
Quantitative validation of stereo disparity and depth estimates.

This is the key addition that makes Module 1 research-grade rather than
a demonstration. In the assignment, there was no ground truth to compare
against — you computed a 3D position and trusted the result. Here we
measure how accurate SGBM actually is by comparing its output to depth
maps derived from CT scanning the same scene.

Metrics used (standard in the stereo literature, e.g. Middlebury benchmark):
  MAE:   Mean Absolute Error — average pixel/mm error
  RMSE:  Root Mean Square Error — penalises large errors more heavily
  >1px:  % of pixels with disparity error > 1 pixel  (strict threshold)
  >2px:  % of pixels with disparity error > 2 pixels (lenient threshold)
  >5mm:  % of pixels with depth error > 5mm

These metrics only count pixels where BOTH the algorithm produced a valid
estimate AND the CT ground truth is reliable (inside the valid_mask).
Comparing apples to apples is essential — the occlusion mask tells us
which regions the CT scanner couldn't measure reliably.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from dataclasses import dataclass


@dataclass
class DisparityMetrics:
    """Evaluation metrics comparing predicted vs ground-truth disparity."""
    mae_px:    float   # mean absolute error in pixels
    rmse_px:   float   # root mean square error in pixels
    pct_1px:   float   # % pixels with error > 1px
    pct_2px:   float   # % pixels with error > 2px
    n_valid:   int     # number of valid pixels evaluated


@dataclass
class DepthMetrics:
    """Evaluation metrics comparing predicted vs ground-truth depth (mm)."""
    mae_mm:    float
    rmse_mm:   float
    pct_5mm:   float   # % pixels with error > 5mm
    pct_10mm:  float   # % pixels with error > 10mm
    n_valid:   int


def evaluate_disparity(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    valid_mask: np.ndarray,
    min_disp: float = 0.0,
) -> DisparityMetrics:
    """
    Compare predicted disparity map against CT ground truth.

    Parameters
    ----------
    predicted:
        SGBM disparity output (float32, pixels). NaN = no estimate.
    ground_truth:
        CT-derived ground-truth disparity (float32, pixels).
    valid_mask:
        Boolean mask: True where GT is reliable (from occlusion image).
    min_disp:
        Ignore predicted values below this (typically invalid SGBM output).
    """
    # A pixel is evaluated only if:
    #   - GT is valid (not in occlusion/non-overlap region)
    #   - SGBM produced a valid estimate (not NaN, not below min_disp)
    pred_valid = ~np.isnan(predicted) & (predicted > min_disp)
    mask = valid_mask & pred_valid

    if mask.sum() == 0:
        return DisparityMetrics(np.nan, np.nan, np.nan, np.nan, 0)

    err = np.abs(predicted[mask] - ground_truth[mask])
    mae   = float(err.mean())
    rmse  = float(np.sqrt((err ** 2).mean()))
    pct_1 = float((err > 1.0).mean() * 100)
    pct_2 = float((err > 2.0).mean() * 100)

    return DisparityMetrics(
        mae_px=mae, rmse_px=rmse,
        pct_1px=pct_1, pct_2px=pct_2,
        n_valid=int(mask.sum()),
    )


def evaluate_depth(
    predicted_depth_mm: np.ndarray,
    ground_truth_depth_mm: np.ndarray,
    valid_mask: np.ndarray,
) -> DepthMetrics:
    """
    Compare predicted depth map against CT ground truth in mm.

    Depth is derived from disparity via the Q matrix and is in the same
    units as the calibration (mm for SERV-CT). The CT ground truth gives
    us absolute depth, so errors here reflect the real-world accuracy
    of the stereo reconstruction.
    """
    pred_valid = np.isfinite(predicted_depth_mm) & (predicted_depth_mm > 0)
    mask = valid_mask & pred_valid

    if mask.sum() == 0:
        return DepthMetrics(np.nan, np.nan, np.nan, np.nan, 0)

    err = np.abs(predicted_depth_mm[mask] - ground_truth_depth_mm[mask])
    return DepthMetrics(
        mae_mm=float(err.mean()),
        rmse_mm=float(np.sqrt((err ** 2).mean())),
        pct_5mm=float((err > 5.0).mean() * 100),
        pct_10mm=float((err > 10.0).mean() * 100),
        n_valid=int(mask.sum()),
    )


def print_validation_summary(
    disp_m: DisparityMetrics,
    depth_m: DepthMetrics,
    frame_id: str = '',
) -> None:
    """Print a formatted validation table."""
    label = f"Frame {frame_id}" if frame_id else "Results"
    print(f"\n{'='*50}")
    print(f"STEREO VALIDATION — {label}")
    print(f"{'='*50}")
    print(f"  Valid pixels evaluated: {disp_m.n_valid:,}")
    print()
    print(f"  DISPARITY")
    print(f"    MAE:       {disp_m.mae_px:.3f} px")
    print(f"    RMSE:      {disp_m.rmse_px:.3f} px")
    print(f"    Err > 1px: {disp_m.pct_1px:.1f}%")
    print(f"    Err > 2px: {disp_m.pct_2px:.1f}%")
    print()
    print(f"  DEPTH")
    print(f"    MAE:       {depth_m.mae_mm:.2f} mm")
    print(f"    RMSE:      {depth_m.rmse_mm:.2f} mm")
    print(f"    Err > 5mm: {depth_m.pct_5mm:.1f}%")
    print(f"{'='*50}")


def plot_validation(
    left_img: np.ndarray,
    predicted_disp: np.ndarray,
    gt_disp: np.ndarray,
    valid_mask: np.ndarray,
    frame_id: str = '',
) -> plt.Figure:
    """
    Four-panel comparison: stereo image, predicted disparity, GT disparity,
    and error map.

    The error map is the most informative panel: bright regions are where
    SGBM disagrees with CT. Errors typically cluster at:
      - Object boundaries (depth discontinuities confuse block matching)
      - Low-texture regions (specular highlights on wet tissue)
      - Occlusion boundaries (right camera can't see behind foreground objects)
    Understanding where the algorithm fails is as important as the headline
    accuracy number.
    """
    import cv2 as _cv2
    pred_valid = ~np.isnan(predicted_disp) & (predicted_disp > 0)
    eval_mask  = valid_mask & pred_valid
    err_map    = np.full_like(predicted_disp, np.nan)
    err_map[eval_mask] = np.abs(predicted_disp[eval_mask] - gt_disp[eval_mask])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()

    # Panel 1: left image
    axes[0].imshow(_cv2.cvtColor(left_img, _cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Left rectified image — frame {frame_id}', fontsize=11)
    axes[0].axis('off')

    # Panel 2: SGBM predicted disparity
    d_range = (gt_disp[valid_mask].min(), gt_disp[valid_mask].max())
    im1 = axes[1].imshow(predicted_disp, cmap='plasma',
                          vmin=d_range[0], vmax=d_range[1])
    axes[1].set_title('SGBM predicted disparity (px)', fontsize=11)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.03, label='px')

    # Panel 3: CT ground-truth disparity
    gt_display = gt_disp.copy()
    gt_display[~valid_mask] = np.nan
    im2 = axes[2].imshow(gt_display, cmap='plasma',
                          vmin=d_range[0], vmax=d_range[1])
    axes[2].set_title('CT ground-truth disparity (px)', fontsize=11)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.03, label='px')

    # Panel 4: absolute error map (clipped at 5px for visibility)
    im3 = axes[3].imshow(err_map, cmap='hot', vmin=0, vmax=5)
    axes[3].set_title('Absolute disparity error (clipped at 5px)', fontsize=11)
    axes[3].axis('off')
    cb = plt.colorbar(im3, ax=axes[3], fraction=0.03, label='px error')
    cb.ax.axhline(1.0, color='cyan', lw=1.5, label='>1px threshold')
    cb.ax.axhline(2.0, color='lime',  lw=1.5)

    valid_pct = eval_mask.mean() * 100
    fig.suptitle(
        f'SERV-CT Validation — Frame {frame_id}  '
        f'({valid_pct:.0f}% of pixels evaluated)',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    return fig


def aggregate_metrics(
    frames_metrics: list[tuple[DisparityMetrics, DepthMetrics]],
) -> tuple[DisparityMetrics, DepthMetrics]:
    """
    Compute weighted-average metrics across multiple frames.

    Each frame contributes proportionally to its number of valid pixels,
    so frames with more valid GT pixels have more weight. This is the
    correct way to aggregate — a simple average of per-frame MAEs would
    weight a frame with 100 valid pixels equally to one with 100,000.
    """
    total_d = sum(d.n_valid for d, _ in frames_metrics)
    total_z = sum(z.n_valid for _, z in frames_metrics)

    w_mae_d  = sum(d.mae_px  * d.n_valid for d, _ in frames_metrics) / total_d
    w_rmse_d = np.sqrt(sum(d.rmse_px**2 * d.n_valid for d, _ in frames_metrics) / total_d)
    w_p1     = sum(d.pct_1px * d.n_valid for d, _ in frames_metrics) / total_d
    w_p2     = sum(d.pct_2px * d.n_valid for d, _ in frames_metrics) / total_d

    w_mae_z  = sum(z.mae_mm  * z.n_valid for _, z in frames_metrics) / total_z
    w_rmse_z = np.sqrt(sum(z.rmse_mm**2 * z.n_valid for _, z in frames_metrics) / total_z)
    w_p5     = sum(z.pct_5mm * z.n_valid for _, z in frames_metrics) / total_z
    w_p10    = sum(z.pct_10mm* z.n_valid for _, z in frames_metrics) / total_z

    return (
        DisparityMetrics(w_mae_d, w_rmse_d, w_p1, w_p2, total_d),
        DepthMetrics(w_mae_z, w_rmse_z, w_p5, w_p10, total_z),
    )
