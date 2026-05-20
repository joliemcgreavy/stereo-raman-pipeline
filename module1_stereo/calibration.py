"""
Stereo camera calibration using checkerboard images.

In the assignment, this was done through MATLAB's Stereo Camera Calibrator GUI —
you loaded images, clicked "Calibrate", and read off the K matrices. Here we
replicate that entire process programmatically using OpenCV.

Why do we need calibration at all?
-----------------------------------
Every real camera introduces two types of distortion:
  - Geometric: the pinhole model assumes straight rays, but lenses bend them
    (barrel/pincushion distortion). The K matrix + distortion coefficients
    describe this mathematically.
  - Stereo geometry: two cameras placed side by side are never perfectly
    parallel or level. The extrinsic parameters (R, T) describe the rigid
    transformation between them. Without knowing R and T precisely, you cannot
    reliably compute depth from disparity.

Calibration solves both problems by showing the cameras a target whose 3D
geometry is known exactly — a checkerboard. Because we know where every corner
is in the real world (e.g. row 0 col 0 is at (0,0,0), row 0 col 1 is at
(5,0,0) for a 5mm square), we can use the observed 2D image positions to
back-calculate all the unknown camera parameters.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class CameraIntrinsics:
    """
    Holds the intrinsic parameters for a single camera.

    K (3x3 matrix):
        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
        fx, fy: focal lengths in pixels (how many pixels per mm in x and y)
        cx, cy: principal point — the pixel coordinate of the optical axis

    dist_coeffs:
        [k1, k2, p1, p2, k3] — radial (k) and tangential (p) distortion.
        The assignment only reported k1, k2 (the two radial terms), which are
        by far the most significant for endoscopic lenses.
    """
    K: np.ndarray
    dist_coeffs: np.ndarray
    reprojection_error: float


@dataclass
class StereoCalibrationResult:
    """
    Full output of stereo calibration.

    left, right: intrinsic parameters for each camera
    R: 3x3 rotation matrix — the orientation of the right camera relative
       to the left. If both cameras faced exactly the same direction, R
       would be the identity matrix.
    T: 3x1 translation vector — how far right camera is from left camera
       in 3D space (the baseline). The baseline is crucial: a wider baseline
       gives better depth resolution at long range.
    E: essential matrix — encodes R and T in a form that directly relates
       corresponding points in the two images (operates on normalised coords).
    F: fundamental matrix — same as E but works with raw pixel coordinates.
       Used to draw epipolar lines.
    Q: 4x4 disparity-to-depth reprojection matrix. This is the same Q matrix
       used in Exercise 3 of the assignment. Multiplying Q by [x, y, d, 1]^T
       gives the homogeneous 3D point [X, Y, Z, W]^T; actual coords are
       (X/W, Y/W, Z/W).
    """
    left: CameraIntrinsics
    right: CameraIntrinsics
    R: np.ndarray
    T: np.ndarray
    E: np.ndarray
    F: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    stereo_reprojection_error: float


def find_checkerboard_corners(
    image_paths: list[Path],
    pattern_size: tuple[int, int],
    square_size_mm: float,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    """
    Detect checkerboard corners in a list of images.

    Parameters
    ----------
    image_paths:
        Paths to calibration images (one camera).
    pattern_size:
        (cols-1, rows-1) — the number of *interior* corners, not squares.
        A 6x8 square checkerboard has a 5x7 interior corner grid.
    square_size_mm:
        Physical size of each square in millimetres. This is what gives the
        result a real-world unit (mm) rather than being dimensionless.
        In the assignment, calibration squares were 5mm and probe squares 1mm.

    Returns
    -------
    obj_points:
        List of 3D point arrays in the checkerboard's coordinate system.
        Each array is identical — it's just the known geometry of the board.
        Every z-coordinate is 0.0 because the board is flat (planar target).
    img_points:
        List of 2D detected corner pixel coordinates, one array per image.
    image_size:
        (width, height) of the images — needed by calibration functions.
    """
    cols, rows = pattern_size

    # Build the ideal 3D object points for one board view.
    # np.mgrid creates a grid; we reshape it to Nx3 float32.
    # Example for a 3x2 pattern: [(0,0,0),(5,0,0),(10,0,0),(0,5,0),...]
    single_board_points = np.zeros((cols * rows, 3), dtype=np.float32)
    single_board_points[:, :2] = (
        np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm
    )

    obj_points = []
    img_points = []
    image_size = None

    # Termination criteria for the sub-pixel corner refinement step below.
    # We stop when either the positional change is < 0.001 px OR we hit
    # 30 iterations — whichever comes first.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        # cv2.findChessboardCorners detects the interior corners.
        # ADAPTIVE_THRESH + NORMALIZE_IMAGE help with uneven lighting —
        # common in endoscopic images where illumination varies across the frame.
        found, corners = cv2.findChessboardCorners(
            gray,
            (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if found:
            # cornerSubPix refines corner positions to sub-pixel accuracy.
            # The initial findChessboardCorners gives integer-pixel estimates;
            # this step fits a local gradient model to get ~0.1 px precision.
            # More precise corners = lower reprojection error = better calibration.
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(single_board_points)
            img_points.append(refined)

    return obj_points, img_points, image_size


def calibrate_single_camera(
    obj_points: list[np.ndarray],
    img_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> CameraIntrinsics:
    """
    Estimate intrinsic parameters for one camera from checkerboard observations.

    OpenCV's calibrateCamera solves a non-linear least squares problem:
    it finds the K matrix and distortion coefficients that minimise the
    reprojection error — the average pixel distance between where corners
    were detected and where the calibrated model predicts they should be.

    A reprojection error below ~0.5 pixels is generally considered good.
    The MATLAB Stereo Camera Calibrator shows this same metric in its GUI.
    """
    error, K, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    return CameraIntrinsics(K=K, dist_coeffs=dist, reprojection_error=error)


def calibrate_stereo(
    left_image_paths: list[Path],
    right_image_paths: list[Path],
    pattern_size: tuple[int, int],
    square_size_mm: float,
) -> StereoCalibrationResult:
    """
    Full stereo calibration pipeline.

    Steps:
      1. Detect corners in all left images
      2. Detect corners in all right images
      3. Keep only image pairs where both cameras detected the board
      4. Calibrate each camera individually (to get initial K estimates)
      5. Run joint stereo calibration to refine K and solve for R, T
      6. Compute rectification transforms and the Q matrix

    The FIX_INTRINSIC flag in step 5 means we use the K matrices from
    step 4 as a starting point but allow small adjustments. This joint
    optimisation ensures the intrinsics and extrinsics are consistent
    with each other, which individual calibration cannot guarantee.
    """
    obj_L, pts_L, size = find_checkerboard_corners(
        left_image_paths, pattern_size, square_size_mm
    )
    obj_R, pts_R, _ = find_checkerboard_corners(
        right_image_paths, pattern_size, square_size_mm
    )

    # Keep only pairs where both cameras found the board.
    # If one camera missed a board view, that pair cannot constrain
    # the relative geometry between the cameras.
    n = min(len(pts_L), len(pts_R))
    obj_points = obj_L[:n]
    img_pts_L = pts_L[:n]
    img_pts_R = pts_R[:n]

    left_intrinsics = calibrate_single_camera(obj_points, img_pts_L, size)
    right_intrinsics = calibrate_single_camera(obj_points, img_pts_R, size)

    flags = (
        cv2.CALIB_FIX_INTRINSIC  # use individual calibrations as fixed starting point
    )

    stereo_error, K_L, d_L, K_R, d_R, R, T, E, F = cv2.stereoCalibrate(
        obj_points,
        img_pts_L,
        img_pts_R,
        left_intrinsics.K,
        left_intrinsics.dist_coeffs,
        right_intrinsics.K,
        right_intrinsics.dist_coeffs,
        size,
        flags=flags,
    )

    # Stereo rectification: compute the rotation matrices (R1, R2) that
    # would need to be applied to each camera to make their image planes
    # perfectly coplanar and parallel. After rectification, corresponding
    # points in the left and right images lie on the same horizontal scanline
    # (same y-coordinate). This is what makes disparity computation trivial:
    # you only need to search horizontally, not across the whole image.
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_L, d_L, K_R, d_R, size, R, T, alpha=0
    )
    # alpha=0 means crop the rectified images so there are no black borders.
    # alpha=1 would keep the full sensor area but include black regions.

    left_result = CameraIntrinsics(K=K_L, dist_coeffs=d_L,
                                    reprojection_error=left_intrinsics.reprojection_error)
    right_result = CameraIntrinsics(K=K_R, dist_coeffs=d_R,
                                     reprojection_error=right_intrinsics.reprojection_error)

    return StereoCalibrationResult(
        left=left_result,
        right=right_result,
        R=R, T=T, E=E, F=F,
        R1=R1, R2=R2, P1=P1, P2=P2,
        Q=Q,
        stereo_reprojection_error=stereo_error,
    )


def print_calibration_summary(result: StereoCalibrationResult) -> None:
    """Print a human-readable summary of calibration results."""
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 60)
    print("STEREO CALIBRATION RESULTS")
    print("=" * 60)

    for label, cam in [("LEFT CAMERA", result.left), ("RIGHT CAMERA", result.right)]:
        print(f"\n{label}")
        print(f"  Reprojection error: {cam.reprojection_error:.4f} px")
        print(f"  Intrinsic matrix K:\n{cam.K}")
        print(f"  Focal length:  fx={cam.K[0,0]:.2f} px, fy={cam.K[1,1]:.2f} px")
        print(f"  Principal point: cx={cam.K[0,2]:.2f}, cy={cam.K[1,2]:.2f}")
        k1, k2 = cam.dist_coeffs[0, 0], cam.dist_coeffs[0, 1]
        print(f"  Radial distortion: k1={k1:.6f}, k2={k2:.6f}")

    print(f"\nSTEREO")
    print(f"  Joint reprojection error: {result.stereo_reprojection_error:.4f} px")
    baseline_mm = np.linalg.norm(result.T)
    print(f"  Baseline (|T|): {baseline_mm:.2f} mm")
    print(f"\n  Q matrix:\n{result.Q}")
    print("=" * 60)
