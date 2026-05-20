"""
Disparity map computation from a rectified stereo image pair.

What is disparity?
------------------
After stereo rectification, corresponding points in the left and right images
lie on the same horizontal scanline. The horizontal pixel offset between a
point in the left image and its match in the right image is called disparity:

    d = x_left - x_right

A large disparity means the object is CLOSE to the camera.
A small disparity means the object is FAR from the camera.
This is why disparity and depth are inversely proportional:

    Z = (f * B) / d

where f is focal length (px), B is baseline (mm), d is disparity (px).

This relationship is exactly what the Q matrix encodes — instead of computing
Z directly, we use Q to project [x, y, d, 1]^T to 3D homogeneous coordinates,
which handles the full projective geometry including the principal point offset.

In the assignment, you manually identified two corresponding points (one in
each rectified image) and computed d = x_left - x_right by hand.
Here we compute a DENSE disparity map — a depth value for every pixel in the
image — using the Semi-Global Block Matching (SGBM) algorithm.
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class DisparityResult:
    disparity_map: np.ndarray    # raw disparity values (float32)
    disparity_visual: np.ndarray  # normalised for display (uint8)


def compute_disparity_sgbm(
    left_rect: np.ndarray,
    right_rect: np.ndarray,
    min_disparity: int = 0,
    num_disparities: int = 128,
    block_size: int = 9,
) -> DisparityResult:
    """
    Compute a dense disparity map using Semi-Global Block Matching.

    SGBM is an improvement over simple block matching. Instead of just
    comparing local patches (block matching), it also penalises large
    disparity changes between adjacent pixels — enforcing spatial smoothness.
    This produces much more complete depth maps on surfaces with low texture
    (like tissue), where pure block matching produces holes.

    Parameters
    ----------
    left_rect, right_rect:
        Rectified stereo image pair (same size, same horizontal scanlines).
    min_disparity:
        Minimum disparity value to search for. 0 is typical unless part of
        the scene is behind the camera (negative disparity).
    num_disparities:
        The search range: how many disparity levels to evaluate. Must be
        divisible by 16. Larger = slower but handles closer objects.
        128 pixels corresponds to roughly 40mm at typical endoscope distances.
    block_size:
        Size of the matching window (must be odd). Larger = smoother but
        loses fine detail. 9 is a good balance for surgical images.
    """
    left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    # P1, P2 are the SGBM smoothness penalty terms.
    # P1 penalises a 1-pixel disparity change between neighbours.
    # P2 penalises larger changes (enforces piecewise-smooth depth).
    # The standard recommendation is P2 = 4 * P1.
    p1 = 8 * block_size ** 2
    p2 = 32 * block_size ** 2

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=1,       # max allowed diff in left-right consistency check
        uniquenessRatio=10,    # reject matches if second-best is within 10%
        speckleWindowSize=100, # small isolated disparity regions to remove
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    # SGBM returns disparity * 16 (fixed-point). Divide to get real disparity.
    disparity_raw = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

    # Replace invalid disparities (< min_disparity) with NaN for downstream use.
    disparity_raw[disparity_raw < min_disparity] = np.nan

    # Create a visual version (normalised to 0-255, colourmap applied).
    valid = ~np.isnan(disparity_raw)
    visual = np.zeros_like(disparity_raw, dtype=np.uint8)
    if valid.any():
        norm = cv2.normalize(
            disparity_raw, None, 0, 255,
            cv2.NORM_MINMAX, dtype=cv2.CV_8U,
            mask=valid.astype(np.uint8),
        )
        visual = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)

    return DisparityResult(disparity_map=disparity_raw, disparity_visual=visual)


def point_disparity(
    left_px: tuple[int, int],
    right_px: tuple[int, int],
) -> float:
    """
    Compute disparity for a single manually-identified point pair.

    This mirrors Exercise 3 of the assignment exactly:
        d = x_left - x_right
    where x is the column index (horizontal pixel coordinate).

    Parameters
    ----------
    left_px:
        (col, row) of the target in the left (rectified) image.
    right_px:
        (col, row) of the same target in the right (rectified) image.
    """
    return float(left_px[0] - right_px[0])
