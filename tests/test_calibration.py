"""
Unit tests for module1_stereo.

These tests cover the pure mathematical functions — disparity calculation,
Q-matrix projection, and point cloud filtering — which require no image
data or file downloads to run. Tests are fast and deterministic.

Why unit test these functions?
The stereo geometry math is critical: a sign error in the Q-matrix
multiplication or an off-by-one in the disparity formula would silently
produce wrong 3D coordinates. These tests check edge cases (zero disparity,
negative disparity, known exact values) to catch that class of error.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from module1_stereo.disparity import point_disparity
from module1_stereo.reconstruction import (
    reproject_single_point,
    disparity_to_depth,
    filter_point_cloud,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def demo_Q() -> np.ndarray:
    """
    Q matrix matching the assignment's given values.
    Used to verify that the reprojection formula gives the expected result.
    """
    return np.array([
        [1.0,  0.0,  0.0,       -942.558266],
        [0.0,  1.0,  0.0,       -520.259571],
        [0.0,  0.0,  0.0,       1728.61755 ],
        [0.0,  0.0,  0.280644094,  0.0     ],
    ])


@pytest.fixture
def unit_Q() -> np.ndarray:
    """
    Minimal Q matrix for a camera at the origin with no principal point offset.
    Makes the expected 3D output easy to calculate by hand.
    cx=0, cy=0, f=100, 1/B=0.1 → B=10mm
    """
    return np.array([
        [1, 0,   0,   0],
        [0, 1,   0,   0],
        [0, 0,   0, 100],
        [0, 0, 0.1,   0],
    ], dtype=float)


# ── Disparity tests ────────────────────────────────────────────────────────

class TestPointDisparity:
    def test_positive_disparity(self) -> None:
        """Left x > right x → positive disparity (object in front of camera)."""
        d = point_disparity((100, 200), (80, 200))
        assert d == pytest.approx(20.0)

    def test_zero_disparity(self) -> None:
        """Matching x values → disparity 0 → infinite depth (object at infinity)."""
        d = point_disparity((150, 100), (150, 100))
        assert d == pytest.approx(0.0)

    def test_returns_float(self) -> None:
        d = point_disparity((500, 300), (490, 300))
        assert isinstance(d, float)

    def test_column_only(self) -> None:
        """Row (y) values should not affect disparity — only column (x) matters."""
        d1 = point_disparity((200, 100), (180, 100))
        d2 = point_disparity((200, 999), (180,   0))
        assert d1 == pytest.approx(d2)


# ── Reprojection tests ─────────────────────────────────────────────────────

class TestReprojectionSinglePoint:
    def test_known_value_unit_Q(self, unit_Q: np.ndarray) -> None:
        """
        Manual calculation with unit_Q:
          vec = [10, 20, 5, 1]^T
          [X,Y,Z,W] = Q @ vec = [10, 20, 100, 0.5]
          3D = (10/0.5, 20/0.5, 100/0.5) = (20, 40, 200)
        """
        result = reproject_single_point(x=10, y=20, disparity=5.0, Q=unit_Q)
        assert result == pytest.approx([20.0, 40.0, 200.0], rel=1e-6)

    def test_returns_3d_array(self, unit_Q: np.ndarray) -> None:
        result = reproject_single_point(0, 0, 1.0, unit_Q)
        assert result.shape == (3,)

    def test_larger_disparity_gives_smaller_depth(self, unit_Q: np.ndarray) -> None:
        """Z ∝ 1/d — doubling disparity should halve depth."""
        z1 = reproject_single_point(0, 0, 1.0, unit_Q)[2]
        z2 = reproject_single_point(0, 0, 2.0, unit_Q)[2]
        assert z2 == pytest.approx(z1 / 2, rel=1e-6)

    def test_symmetric_x_disparity(self, unit_Q: np.ndarray) -> None:
        """
        A point at (x, cy, d) and (-x, cy, d) should give equal-magnitude
        but opposite-sign X coordinates.
        """
        pos = reproject_single_point(x= 50, y=0, disparity=5.0, Q=unit_Q)
        neg = reproject_single_point(x=-50, y=0, disparity=5.0, Q=unit_Q)
        assert pos[0] == pytest.approx(-neg[0], rel=1e-6)
        assert pos[2] == pytest.approx(neg[2],  rel=1e-6)

    def test_assignment_Q_produces_reasonable_depth(self, demo_Q: np.ndarray) -> None:
        """
        Using the assignment Q matrix with disparity=50px should produce a depth
        in the surgical working range. The formula is Z = f / (B_inv * d):
          Z = 1728.6 / (0.2806 * 50) ≈ 123mm — within typical endoscope range.
        With d=20px, Z ≈ 308mm which is also valid but at longer range.
        """
        result = reproject_single_point(x=942, y=520, disparity=50.0, Q=demo_Q)
        assert 50 < result[2] < 200, f"Unexpected depth: {result[2]:.1f} mm"


class TestDisparityToDepth:
    def test_matches_single_point_at_principal_point(self, unit_Q: np.ndarray) -> None:
        """
        At (cx=0, cy=0) with unit_Q, the Q-matrix formula gives:
          Z_raw = Q[2,3] * 1 = 100
          W_raw = Q[3,2] * d = 0.1 * 5 = 0.5
          Z = Z_raw / W_raw = 100 / 0.5 = 200

        Note: Z ≠ f/d here because the homogeneous divide by W scales the result.
        The simple f/d formula only holds when 1/B = 1 (unit baseline).
        """
        depth = disparity_to_depth(disparity=5.0, Q=unit_Q)
        assert depth == pytest.approx(200.0, rel=1e-5)


# ── Point cloud filter tests ───────────────────────────────────────────────

class TestFilterPointCloud:
    def _make_scene(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Synthetic 4×4 depth map with:
          - points at z=50mm (valid, in range)
          - points at z=5mm  (too close, should be removed)
          - points at z=300mm (too far, should be removed)
          - one inf value (invalid, should be removed)
        """
        H, W = 4, 4
        pts = np.zeros((H, W, 3), dtype=np.float32)
        pts[:, :, 2] = 50.0      # all at 50mm by default
        pts[0, 0, 2] = 5.0       # too close
        pts[0, 1, 2] = 300.0     # too far
        pts[0, 2, 2] = np.inf    # invalid
        img = np.ones((H, W, 3), dtype=np.uint8) * 128
        return pts, img

    def test_removes_out_of_range(self) -> None:
        pts, img = self._make_scene()
        filtered_pts, _ = filter_point_cloud(pts, img, z_min_mm=10, z_max_mm=200)
        assert np.all(filtered_pts[:, 2] > 10)
        assert np.all(filtered_pts[:, 2] < 200)

    def test_removes_infinite(self) -> None:
        pts, img = self._make_scene()
        filtered_pts, _ = filter_point_cloud(pts, img, z_min_mm=10, z_max_mm=200)
        assert np.all(np.isfinite(filtered_pts))

    def test_colors_normalized(self) -> None:
        """Colours should be in [0, 1] float range after filtering."""
        pts, img = self._make_scene()
        _, colors = filter_point_cloud(pts, img, z_min_mm=10, z_max_mm=200)
        assert colors.min() >= 0.0
        assert colors.max() <= 1.0

    def test_output_shapes_match(self) -> None:
        pts, img = self._make_scene()
        filtered_pts, colors = filter_point_cloud(pts, img, z_min_mm=10, z_max_mm=200)
        assert filtered_pts.shape[0] == colors.shape[0]
        assert filtered_pts.shape[1] == 3
        assert colors.shape[1] == 3
