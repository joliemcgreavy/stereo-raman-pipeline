"""
3D reconstruction from disparity using the Q matrix.

This is the direct Python equivalent of Exercise 3 in the assignment:
    Q * [x, y, d, 1]^T = [X, Y, Z, W]^T
    3D point = (X/W, Y/W, Z/W)

The Q matrix encodes the complete projective geometry of the stereo rig.
Its structure (for a standard horizontal stereo setup) is:

    Q = [[1, 0,    0,    -cx         ],
         [0, 1,    0,    -cy         ],
         [0, 0,    0,     f          ],
         [0, 0, -1/B,   (cx-cx')/B  ]]

where cx, cy = principal point of left camera,
      f      = focal length,
      B      = baseline (in same units as you want your 3D output),
      cx'    = principal point x of right camera.

OpenCV's stereoRectify computes Q automatically from the calibration
result, so you don't need to fill in those values manually.
"""

import cv2
import numpy as np


def reproject_to_3d(
    disparity_map: np.ndarray,
    Q: np.ndarray,
) -> np.ndarray:
    """
    Convert a full disparity map to a 3D point cloud using the Q matrix.

    cv2.reprojectImageTo3D applies the Q matrix multiplication to every
    pixel simultaneously. This is the vectorised version of what you did
    by hand for a single point in Exercise 3.

    Returns an (H, W, 3) array of (X, Y, Z) coordinates in mm.
    Pixels with invalid disparity will have Z = inf or very large values;
    filter these out before further processing.
    """
    points_3d = cv2.reprojectImageTo3D(disparity_map, Q, handleMissingValues=True)
    return points_3d


def reproject_single_point(
    x: int,
    y: int,
    disparity: float,
    Q: np.ndarray,
) -> np.ndarray:
    """
    Reproject a single 2D point + disparity to 3D.

    This exactly mirrors the manual calculation from Exercise 3:
      vec = [x, y, d, 1]^T
      [X, Y, Z, W] = Q @ vec
      3D = (X/W, Y/W, Z/W)

    Parameters
    ----------
    x, y:   pixel column and row in the left rectified image
    disparity: d = x_left - x_right
    Q:      4x4 disparity-to-depth matrix from stereo calibration
    """
    vec = np.array([x, y, disparity, 1.0], dtype=np.float64)
    result = Q @ vec                        # matrix-vector multiplication
    coords_3d = result[:3] / result[3]      # homogeneous divide
    return coords_3d


def disparity_to_depth(disparity: float, Q: np.ndarray) -> float:
    """
    Extract just the Z (depth) component from a disparity value at the
    image centre — useful for a quick sanity-check of calibration quality.

    At the principal point (cx, cy), the Q matrix formula simplifies to:
        Z = f / d   (where f is the focal length in pixels)
    This is the simplest form of the stereo depth equation.
    """
    cx = -Q[0, 3]
    cy = -Q[1, 3]
    return float(reproject_single_point(int(cx), int(cy), disparity, Q)[2])


def filter_point_cloud(
    points_3d: np.ndarray,
    color_image: np.ndarray,
    z_min_mm: float = 10.0,
    z_max_mm: float = 200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove invalid and out-of-range points from the 3D point cloud.

    Parameters
    ----------
    points_3d:    (H, W, 3) array from reproject_to_3d
    color_image:  (H, W, 3) BGR image for colour assignment to each point
    z_min_mm:     discard points closer than this (likely noise)
    z_max_mm:     discard points farther than this (invalid disparities often
                  produce very large Z values)

    Returns
    -------
    pts:    (N, 3) valid 3D points
    colors: (N, 3) corresponding RGB colours (0-1 float)
    """
    Z = points_3d[:, :, 2]
    mask = (Z > z_min_mm) & (Z < z_max_mm) & np.isfinite(Z)

    pts = points_3d[mask]
    bgr = color_image[mask].astype(np.float32) / 255.0
    rgb = bgr[:, ::-1]  # BGR → RGB

    return pts, rgb
