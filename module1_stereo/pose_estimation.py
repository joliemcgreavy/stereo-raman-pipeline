"""
Single-image pose estimation: estimate camera-to-object distance.

This corresponds to Exercise 2 of the assignment — given a rectified image
of a checkerboard (here: at the tip of the Raman probe), estimate how far
the checkerboard is from the camera.

How it works
------------
solvePnP (Perspective-n-Point) solves for the 6-DOF pose of a known 3D
object given its 2D projections in an image. "Known 3D object" means we
know the geometry of the checkerboard exactly (square size is given).

The output is:
  rvec: rotation vector (Rodrigues form) — 3 numbers encoding the 3D
        orientation of the board relative to the camera
  tvec: translation vector — the 3D position of the board origin relative
        to the camera. The Z component of tvec IS the depth (distance along
        the optical axis), and |tvec| is the Euclidean distance.

Why can we do this from a SINGLE image?
----------------------------------------
With a single camera you cannot recover depth from a single point — it's
ambiguous (the point could be close and small, or far and large). But with
a planar target whose geometry you know exactly, you have multiple points
with known 3D relationships. This over-constrains the problem enough to
recover the full pose. This is fundamentally different from stereo depth,
which uses two images of the same unknown scene.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PoseEstimationResult:
    rvec: np.ndarray
    tvec: np.ndarray
    distance_mm: float
    reprojection_error_px: float


def estimate_checkerboard_pose(
    image_path: Path,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    pattern_size: tuple[int, int],
    square_size_mm: float,
) -> PoseEstimationResult | None:
    """
    Estimate the 6-DOF pose of a checkerboard in a single image.

    Parameters
    ----------
    image_path:
        Path to a rectified (undistorted) image. Because the image is already
        rectified, we pass zero distortion to solvePnP — the undistortion has
        already been applied to the pixel coordinates.
    K:
        Intrinsic matrix from stereo calibration.
    dist_coeffs:
        Distortion coefficients. Pass np.zeros((5,1)) if the image is
        already rectified/undistorted.
    pattern_size:
        (cols-1, rows-1) interior corners.
    square_size_mm:
        Physical square size.

    Returns None if the checkerboard is not found.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cols, rows = pattern_size

    found, corners = cv2.findChessboardCorners(
        gray,
        (cols, rows),
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )

    if not found:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Build the known 3D positions of the checkerboard corners.
    obj_points = np.zeros((cols * rows, 3), dtype=np.float32)
    obj_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    # solvePnP finds rvec and tvec such that the projected 3D points best
    # match the observed 2D corners.
    # SOLVEPNP_ITERATIVE uses Levenberg-Marquardt optimisation — the same
    # underlying method MATLAB's estimateWorldCameraPose uses.
    success, rvec, tvec = cv2.solvePnP(
        obj_points, corners, K, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return None

    # tvec is [tx, ty, tz] in mm (because square_size_mm was in mm).
    # The Euclidean distance from camera to board origin is |tvec|.
    distance_mm = float(np.linalg.norm(tvec))

    # Measure reprojection error: project the 3D points back into the image
    # using the estimated pose, then measure pixel distance from detections.
    projected, _ = cv2.projectPoints(obj_points, rvec, tvec, K, dist_coeffs)
    error = float(np.mean(np.linalg.norm(corners - projected, axis=2)))

    return PoseEstimationResult(
        rvec=rvec,
        tvec=tvec,
        distance_mm=distance_mm,
        reprojection_error_px=error,
    )


def draw_pose_axes(
    image: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    result: PoseEstimationResult,
    axis_length_mm: float = 10.0,
) -> np.ndarray:
    """
    Draw XYZ coordinate axes on the image at the checkerboard origin.

    This is a standard sanity-check visualisation: if the axes look right
    (pointing in sensible directions relative to the board), the pose is correct.
    Red=X, Green=Y, Blue=Z (Z points toward the camera).
    """
    img = image.copy()
    origin = np.float32([[0, 0, 0]])
    axes = np.float32([
        [axis_length_mm, 0, 0],
        [0, axis_length_mm, 0],
        [0, 0, -axis_length_mm],  # negative Z points away from camera
    ])
    pts_origin, _ = cv2.projectPoints(origin, result.rvec, result.tvec, K, dist_coeffs)
    pts_axes, _ = cv2.projectPoints(axes, result.rvec, result.tvec, K, dist_coeffs)

    o = tuple(pts_origin[0].ravel().astype(int))
    cv2.line(img, o, tuple(pts_axes[0].ravel().astype(int)), (0, 0, 255), 3)  # X red
    cv2.line(img, o, tuple(pts_axes[1].ravel().astype(int)), (0, 255, 0), 3)  # Y green
    cv2.line(img, o, tuple(pts_axes[2].ravel().astype(int)), (255, 0, 0), 3)  # Z blue
    return img
