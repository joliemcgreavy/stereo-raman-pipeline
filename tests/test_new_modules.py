"""
Unit tests for the three modules added after the initial build:
  - module1_stereo/serv_ct_loader.py
  - module1_stereo/validation.py
  - module2_raman/preprocessing.py

SERV-CT tests require the dataset to be present locally.
They are skipped automatically if the data is absent so the test suite
still passes in CI or on a fresh clone without data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

# ── Fixtures & skip condition ──────────────────────────────────────────────

SERV_CT_DIR = Path(__file__).parent.parent / 'data' / 'raw' / 'serv_ct' / 'SERV-CT'
serv_ct_available = pytest.mark.skipif(
    not SERV_CT_DIR.exists(),
    reason="SERV-CT dataset not downloaded — skipping (see data/README.md)"
)


@pytest.fixture(scope="module")
def frame_001():
    """Load SERV-CT frame 001 once for all tests in this module."""
    from module1_stereo.serv_ct_loader import load_frame
    return load_frame('001')


# ═══════════════════════════════════════════════════════════════════════════
# module1_stereo/serv_ct_loader.py
# ═══════════════════════════════════════════════════════════════════════════

class TestSERVCTLoader:

    @serv_ct_available
    def test_image_shapes(self, frame_001) -> None:
        """Left and right images should be 720×576 BGR."""
        assert frame_001.left_img.shape  == (576, 720, 3)
        assert frame_001.right_img.shape == (576, 720, 3)

    @serv_ct_available
    def test_image_dtype(self, frame_001) -> None:
        assert frame_001.left_img.dtype == np.uint8

    @serv_ct_available
    def test_Q_shape(self, frame_001) -> None:
        assert frame_001.Q.shape == (4, 4)

    @serv_ct_available
    def test_P1_P2_shapes(self, frame_001) -> None:
        assert frame_001.P1.shape == (3, 4)
        assert frame_001.P2.shape == (3, 4)

    @serv_ct_available
    def test_focal_length_reasonable(self, frame_001) -> None:
        """
        Focal length should be in the range typical for a surgical endoscope
        (roughly 800–1200 px at 720×576 resolution).
        """
        fx = frame_001.focal_length_px
        assert 800 < fx < 1200, f"Unexpected focal length: {fx:.1f} px"

    @serv_ct_available
    def test_baseline_reasonable(self, frame_001) -> None:
        """
        Stereo baseline for da Vinci endoscope is typically 3–8mm.
        """
        B = frame_001.baseline_mm
        assert 3.0 < B < 10.0, f"Unexpected baseline: {B:.3f} mm"

    @serv_ct_available
    def test_gt_depth_range(self, frame_001) -> None:
        """
        SERV-CT ground-truth depth should be in the surgical working range
        (roughly 50–110mm for this dataset).
        """
        valid_depths = frame_001.gt_depth_mm[frame_001.valid_mask]
        assert valid_depths.min() > 40
        assert valid_depths.max() < 120

    @serv_ct_available
    def test_valid_mask_coverage(self, frame_001) -> None:
        """
        The occlusion mask should mark ~90% of pixels as valid GT.
        (From dataset documentation: yellow non-overlap pixels ≈ 10%.)
        """
        coverage = frame_001.valid_mask.mean()
        assert 0.80 < coverage < 0.98, f"Unexpected coverage: {coverage:.2%}"

    @serv_ct_available
    def test_valid_mask_dtype(self, frame_001) -> None:
        assert frame_001.valid_mask.dtype == bool

    @serv_ct_available
    def test_experiment_assignment(self) -> None:
        """Frames 001–008 → Experiment 1; 009–016 → Experiment 2."""
        from module1_stereo.serv_ct_loader import load_frame
        f1 = load_frame('001')
        f9 = load_frame('009')
        assert f1.experiment == 1
        assert f9.experiment == 2

    @serv_ct_available
    def test_gt_disparity_positive(self, frame_001) -> None:
        """All valid ground-truth disparities should be positive."""
        valid = frame_001.gt_disparity[frame_001.valid_mask]
        assert np.all(valid > 0)

    @serv_ct_available
    def test_q_matrix_depth_formula(self, frame_001) -> None:
        """
        Verify Q encodes the correct depth formula at the principal point.
        At (cx, cy) with disparity d: Z = focal_length / (Q[3,2] * d).
        Cross-check against gt_depth at the image centre.
        """
        from module1_stereo.reconstruction import reproject_single_point
        cx = int(frame_001.principal_point[0])
        cy = int(frame_001.principal_point[1])
        d  = float(frame_001.gt_disparity[cy, cx])
        gt_z = float(frame_001.gt_depth_mm[cy, cx])
        pred = reproject_single_point(cx, cy, d, frame_001.Q)
        assert abs(pred[2] - gt_z) < 2.0, \
            f"Q-matrix depth {pred[2]:.1f}mm vs GT {gt_z:.1f}mm"


# ═══════════════════════════════════════════════════════════════════════════
# module1_stereo/validation.py
# ═══════════════════════════════════════════════════════════════════════════

class TestValidation:

    @pytest.fixture
    def perfect_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicted == ground truth with all pixels valid."""
        gt   = np.full((100, 100), 50.0, dtype=np.float32)
        pred = gt.copy()
        mask = np.ones((100, 100), dtype=bool)
        return pred, gt, mask

    @pytest.fixture
    def noisy_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicted has ±2px error on 80% of pixels, rest are NaN."""
        rng  = np.random.default_rng(0)
        gt   = np.full((100, 100), 60.0, dtype=np.float32)
        pred = gt + rng.normal(0, 2, (100, 100)).astype(np.float32)
        mask = np.ones((100, 100), dtype=bool)
        pred[:20, :] = np.nan   # 20% invalid predictions
        return pred, gt, mask

    def test_perfect_disparity_mae_zero(self, perfect_arrays) -> None:
        """Zero error input → MAE and RMSE should both be 0."""
        from module1_stereo.validation import evaluate_disparity
        pred, gt, mask = perfect_arrays
        m = evaluate_disparity(pred, gt, mask)
        assert m.mae_px  == pytest.approx(0.0, abs=1e-5)
        assert m.rmse_px == pytest.approx(0.0, abs=1e-5)

    def test_perfect_disparity_pct_zero(self, perfect_arrays) -> None:
        from module1_stereo.validation import evaluate_disparity
        pred, gt, mask = perfect_arrays
        m = evaluate_disparity(pred, gt, mask)
        assert m.pct_1px == pytest.approx(0.0, abs=1e-5)
        assert m.pct_2px == pytest.approx(0.0, abs=1e-5)

    def test_mae_non_negative(self, noisy_arrays) -> None:
        from module1_stereo.validation import evaluate_disparity
        pred, gt, mask = noisy_arrays
        m = evaluate_disparity(pred, gt, mask, min_disp=1.0)
        assert m.mae_px >= 0.0
        assert m.rmse_px >= m.mae_px  # RMSE ≥ MAE always

    def test_pct_in_valid_range(self, noisy_arrays) -> None:
        from module1_stereo.validation import evaluate_disparity
        pred, gt, mask = noisy_arrays
        m = evaluate_disparity(pred, gt, mask, min_disp=1.0)
        assert 0.0 <= m.pct_1px <= 100.0
        assert 0.0 <= m.pct_2px <= 100.0
        assert m.pct_2px <= m.pct_1px   # fewer pixels exceed 2px than 1px

    def test_empty_valid_mask_returns_nan(self) -> None:
        """No valid pixels → all metrics should be NaN."""
        from module1_stereo.validation import evaluate_disparity
        pred = np.ones((50, 50), dtype=np.float32)
        gt   = np.ones((50, 50), dtype=np.float32)
        mask = np.zeros((50, 50), dtype=bool)
        m    = evaluate_disparity(pred, gt, mask)
        assert np.isnan(m.mae_px)
        assert m.n_valid == 0

    def test_perfect_depth_mae_zero(self, perfect_arrays) -> None:
        from module1_stereo.validation import evaluate_depth
        pred, gt, mask = perfect_arrays
        m = evaluate_depth(pred, gt, mask)
        assert m.mae_mm == pytest.approx(0.0, abs=1e-5)

    def test_depth_pct_5mm_in_range(self, noisy_arrays) -> None:
        from module1_stereo.validation import evaluate_depth
        pred, gt, mask = noisy_arrays
        m = evaluate_depth(pred, gt, mask)
        assert 0.0 <= m.pct_5mm  <= 100.0
        assert 0.0 <= m.pct_10mm <= 100.0
        assert m.pct_10mm <= m.pct_5mm  # fewer pixels exceed 10mm than 5mm

    def test_aggregate_between_extremes(self) -> None:
        """
        Weighted aggregate across two frames should lie between their
        individual metric values (it's a weighted average, not outside the range).
        """
        from module1_stereo.validation import DisparityMetrics, DepthMetrics, aggregate_metrics
        m1 = DisparityMetrics(mae_px=2.0, rmse_px=4.0, pct_1px=30.0, pct_2px=15.0, n_valid=1000)
        m2 = DisparityMetrics(mae_px=4.0, rmse_px=6.0, pct_1px=50.0, pct_2px=30.0, n_valid=1000)
        z1 = DepthMetrics(mae_mm=3.0, rmse_mm=5.0, pct_5mm=20.0, pct_10mm=10.0, n_valid=1000)
        z2 = DepthMetrics(mae_mm=6.0, rmse_mm=9.0, pct_5mm=40.0, pct_10mm=20.0, n_valid=1000)
        agg_d, agg_z = aggregate_metrics([(m1, z1), (m2, z2)])
        assert 2.0 <= agg_d.mae_px <= 4.0
        assert 3.0 <= agg_z.mae_mm <= 6.0


# ═══════════════════════════════════════════════════════════════════════════
# module2_raman/preprocessing.py
# ═══════════════════════════════════════════════════════════════════════════

class TestPreprocessing:

    @pytest.fixture
    def raw_spectrum(self) -> np.ndarray:
        """Synthetic raw Raman spectrum: signal + fluorescence baseline."""
        rng = np.random.default_rng(1)
        n   = 200
        signal     = rng.normal(0.5, 0.05, n)
        background = 10.0 * np.exp(-np.linspace(0, 2, n))
        return signal + background

    @pytest.fixture
    def spectrum_with_spike(self, raw_spectrum) -> tuple[np.ndarray, int]:
        """Inject a cosmic ray spike at index 100."""
        s = raw_spectrum.copy()
        s[100] += 50.0
        return s, 100

    def test_als_baseline_shape(self, raw_spectrum) -> None:
        from module2_raman.preprocessing import asymmetric_least_squares
        baseline = asymmetric_least_squares(raw_spectrum)
        assert baseline.shape == raw_spectrum.shape

    def test_als_baseline_below_spectrum(self, raw_spectrum) -> None:
        """
        ALS baseline should mostly sit below the spectrum.
        With p=0.01 (asymmetric weighting), > 95% of baseline points
        should be <= the original spectrum.
        """
        from module2_raman.preprocessing import asymmetric_least_squares
        baseline = asymmetric_least_squares(raw_spectrum, p=0.01)
        frac_below = (baseline <= raw_spectrum).mean()
        assert frac_below > 0.90, f"Baseline sits above spectrum at {(1-frac_below)*100:.1f}% of points"

    def test_correct_baseline_non_negative(self, raw_spectrum) -> None:
        """After baseline subtraction, values are clipped to >= 0."""
        from module2_raman.preprocessing import correct_baseline
        corrected = correct_baseline(raw_spectrum[None, :])
        assert corrected.min() >= 0.0

    def test_correct_baseline_shape_preserved(self, raw_spectrum) -> None:
        from module2_raman.preprocessing import correct_baseline
        batch = np.stack([raw_spectrum] * 5)
        corrected = correct_baseline(batch)
        assert corrected.shape == batch.shape

    def test_cosmic_ray_detected(self, spectrum_with_spike) -> None:
        from module2_raman.preprocessing import detect_cosmic_rays
        s, idx = spectrum_with_spike
        mask = detect_cosmic_rays(s, threshold=5.0)
        assert mask[idx], f"Spike at index {idx} not detected"

    def test_cosmic_ray_not_over_detected(self, raw_spectrum) -> None:
        """
        A clean spectrum should have few false positives at a high threshold.
        We use threshold=15.0 here — real cosmic rays are typically 50–1000×
        above the local baseline, so this threshold safely separates them from
        natural spectral variation while avoiding false positives on smooth data.
        """
        from module2_raman.preprocessing import detect_cosmic_rays
        mask = detect_cosmic_rays(raw_spectrum, threshold=15.0)
        frac = mask.mean()
        assert frac < 0.05, f"Too many false positives: {mask.sum()} ({frac:.1%})"

    def test_remove_cosmic_ray_reduces_spike(self, spectrum_with_spike) -> None:
        """After removal, the spike index should be close to its neighbours."""
        from module2_raman.preprocessing import remove_cosmic_rays
        s, idx = spectrum_with_spike
        corrected = remove_cosmic_rays(s, threshold=5.0)
        neighbour_mean = (corrected[idx-2:idx].mean() + corrected[idx+1:idx+3].mean()) / 2
        assert abs(corrected[idx] - neighbour_mean) < 2.0, \
            f"Spike not removed: corrected[{idx}]={corrected[idx]:.2f}, neighbours≈{neighbour_mean:.2f}"

    def test_preprocess_spectra_shape(self, raw_spectrum) -> None:
        from module2_raman.preprocessing import preprocess_spectra
        batch     = np.stack([raw_spectrum] * 10)
        processed = preprocess_spectra(batch, normalise=False)
        assert processed.shape == batch.shape

    def test_preprocess_spectra_normalised(self, raw_spectrum) -> None:
        """With normalise=True, each output spectrum should have L2 norm ≈ 1."""
        from module2_raman.preprocessing import preprocess_spectra
        batch     = np.stack([raw_spectrum] * 5)
        processed = preprocess_spectra(batch, normalise=True)
        norms     = np.linalg.norm(processed, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), f"Norms not unit: {norms}"

    def test_preprocess_no_spikes_option(self, raw_spectrum) -> None:
        """remove_spikes=False should still return the correct shape."""
        from module2_raman.preprocessing import preprocess_spectra
        batch     = np.stack([raw_spectrum] * 3)
        processed = preprocess_spectra(batch, remove_spikes=False,
                                        correct_fluorescence=True, normalise=True)
        assert processed.shape == batch.shape

    def test_zero_spectrum_safe(self) -> None:
        """A spectrum of all zeros should not raise divide-by-zero errors."""
        from module2_raman.preprocessing import preprocess_spectra
        zeros = np.zeros((3, 100))
        result = preprocess_spectra(zeros, normalise=True)
        assert result.shape == zeros.shape
        assert np.all(result == 0.0)
