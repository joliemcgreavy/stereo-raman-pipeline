"""
SERV-CT dataset loader for Module 1.

SERV-CT (Stereo-Endoscopic Reconstruction Validation based on CT) is an
open dataset from UCL's Wellcome/EPSRC Centre for Interventional and
Surgical Sciences (WEISS). It provides 16 rectified stereo image pairs
from ex vivo porcine tissue filmed with a da Vinci™ endoscope, with
ground-truth depth and disparity maps derived from cone-beam CT.

Paper: Psychogyios et al., Medical Image Analysis (2022)
Source: https://rdr.ucl.ac.uk/articles/dataset/26352199

What makes this dataset valuable for Module 1:
  - Images are ALREADY rectified — corresponding points are on the same
    horizontal scanline. We skip the rectification step and go straight
    to disparity computation.
  - Calibration JSON files contain P1, P2, and Q directly — no need to
    run stereo calibration. The Q matrix is what we use for 3D projection
    (same as Exercise 3 of the assignment).
  - Ground-truth depth and disparity from CT allow us to VALIDATE our SGBM
    results quantitatively, not just visually. This is the key addition
    over the assignment — we can measure how accurate the pipeline is.

Dataset structure:
  SERV-CT/
  ├── Experiment_1/  (frames 001–008, straight endoscope)
  │   ├── Left_rectified/        720×576 colour PNG
  │   ├── Right_rectified/       720×576 colour PNG
  │   ├── Rectified_calibration/ JSON with P1, P2, Q
  │   └── Reference_CT/
  │       ├── DepthL/      16-bit PNG, depth_mm = pixel_value / 256
  │       ├── Disparity/   16-bit PNG, disparity_px = pixel_value / 256
  │       └── OcclusionL/  colour PNG — yellow pixels are invalid GT
  └── Experiment_2/  (frames 009–016, 30° endoscope)
      └── (same structure)
"""

from __future__ import annotations

import json
import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path

# Default location relative to project root
DEFAULT_DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw' / 'serv_ct' / 'SERV-CT'

# Yellow in BGR = [0, 255, 255] — marks invalid / non-overlap pixels
INVALID_COLOUR_BGR = np.array([0, 255, 255], dtype=np.uint8)


@dataclass
class SERVCTFrame:
    """
    A single SERV-CT stereo frame with images, calibration, and ground truth.

    frame_id:       zero-padded string, e.g. '001'
    experiment:     1 or 2
    left_img:       (576, 720, 3) uint8 BGR, left rectified image
    right_img:      (576, 720, 3) uint8 BGR, right rectified image
    Q:              (4, 4) float64 — disparity-to-depth projection matrix
    P1:             (3, 4) float64 — left camera projection matrix
    P2:             (3, 4) float64 — right camera projection matrix
    gt_depth_mm:    (576, 720) float32 — ground-truth depth in mm (CT-derived)
    gt_disparity:   (576, 720) float32 — ground-truth disparity in pixels
    valid_mask:     (576, 720) bool   — True where GT is reliable
    """
    frame_id:    str
    experiment:  int
    left_img:    np.ndarray
    right_img:   np.ndarray
    Q:           np.ndarray
    P1:          np.ndarray
    P2:          np.ndarray
    gt_depth_mm: np.ndarray
    gt_disparity: np.ndarray
    valid_mask:  np.ndarray

    @property
    def focal_length_px(self) -> float:
        """fx from Q matrix — Q[2,3] = focal length."""
        return float(self.Q[2, 3])

    @property
    def baseline_mm(self) -> float:
        """
        Stereo baseline in mm, derived from Q.
        Q[3,2] = 1/B where B is the baseline in mm.
        """
        return float(1.0 / self.Q[3, 2])

    @property
    def principal_point(self) -> tuple[float, float]:
        """(cx, cy) from Q matrix."""
        return float(-self.Q[0, 3]), float(-self.Q[1, 3])


def _parse_matrix(d: dict) -> np.ndarray:
    """Parse an OpenCV-format matrix dict from JSON into a numpy array."""
    rows = d['rows']
    cols = d['cols']
    return np.array(d['data'], dtype=np.float64).reshape(rows, cols)


def _load_calibration(json_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load P1, P2, Q from a SERV-CT calibration JSON file."""
    with open(json_path) as f:
        cal = json.load(f)
    P1 = _parse_matrix(cal['P1'])
    P2 = _parse_matrix(cal['P2'])
    Q  = _parse_matrix(cal['Q'])
    return P1, P2, Q


def _load_gt_depth(depth_png: Path) -> np.ndarray:
    """
    Load a ground-truth depth map from a 16-bit PNG.

    The dataset stores depth_mm * 256 as uint16, giving sub-millimetre
    precision in a lossless integer format. Dividing by 256 recovers
    the actual depth in millimetres.
    """
    raw = cv2.imread(str(depth_png), cv2.IMREAD_UNCHANGED)
    return (raw / 256.0).astype(np.float32)


def _load_gt_disparity(disp_png: Path) -> np.ndarray:
    """
    Load a ground-truth disparity map from a 16-bit PNG.

    Same encoding as depth: stored as disparity_px * 256 for sub-pixel
    precision. Dividing by 256 gives the actual disparity in pixels.
    """
    raw = cv2.imread(str(disp_png), cv2.IMREAD_UNCHANGED)
    return (raw / 256.0).astype(np.float32)


def _load_valid_mask(occlusion_png: Path) -> np.ndarray:
    """
    Derive a valid-GT mask from the SERV-CT occlusion image.

    The occlusion image marks three types of invalid pixel:
      - Yellow [0, 255, 255] BGR: no overlap between left and right image
      - Blue   [0, 0, 255]   BGR: outside the reference surface from CT
      - Red    [0, 0, 255]   BGR: not visible in the right image

    In practice, only yellow (non-overlap, ~10% of pixels) appears
    extensively. We treat all coloured pixels as invalid and only trust
    the ground truth where the occlusion image is NOT yellow.

    Returns a boolean array: True = valid GT, False = ignore.
    """
    occ = cv2.imread(str(occlusion_png))
    yellow = np.all(occ == INVALID_COLOUR_BGR, axis=2)
    return ~yellow


def load_frame(
    frame_id: str,
    data_dir: Path | str | None = None,
) -> SERVCTFrame:
    """
    Load a single SERV-CT frame by its ID string (e.g. '001', '009').

    Frames 001–008 are in Experiment_1 (straight endoscope).
    Frames 009–016 are in Experiment_2 (30° angled endoscope).

    Parameters
    ----------
    frame_id:
        Zero-padded 3-digit string: '001' through '016'.
    data_dir:
        Path to the SERV-CT/ folder. Defaults to data/raw/serv_ct/SERV-CT/
        relative to the project root.
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    n = int(frame_id)
    experiment = 1 if n <= 8 else 2
    exp_dir = data_dir / f'Experiment_{experiment}'

    # Images
    left_img  = cv2.imread(str(exp_dir / 'Left_rectified'  / f'{frame_id}.png'))
    right_img = cv2.imread(str(exp_dir / 'Right_rectified' / f'{frame_id}.png'))
    if left_img is None or right_img is None:
        raise FileNotFoundError(
            f"Could not load frame {frame_id} from {exp_dir}.\n"
            f"Run: cd data/raw/serv_ct && curl -L -o SERV-CT.zip "
            f"https://ndownloader.figshare.com/files/47857471 && unzip SERV-CT.zip"
        )

    # Calibration
    cal_path = exp_dir / 'Rectified_calibration' / f'{frame_id}.json'
    P1, P2, Q = _load_calibration(cal_path)

    # Ground truth (CT reference)
    ct_dir = exp_dir / 'Reference_CT'
    gt_depth    = _load_gt_depth(ct_dir / 'DepthL'      / f'{frame_id}.png')
    gt_disparity = _load_gt_disparity(ct_dir / 'Disparity' / f'{frame_id}.png')
    valid_mask  = _load_valid_mask(ct_dir / 'OcclusionL' / f'{frame_id}.png')

    return SERVCTFrame(
        frame_id=frame_id,
        experiment=experiment,
        left_img=left_img,
        right_img=right_img,
        Q=Q,
        P1=P1,
        P2=P2,
        gt_depth_mm=gt_depth,
        gt_disparity=gt_disparity,
        valid_mask=valid_mask,
    )


def load_all_frames(data_dir: Path | str | None = None) -> list[SERVCTFrame]:
    """Load all 16 SERV-CT frames. Returns them in order 001–016."""
    frames = []
    for n in range(1, 17):
        frame_id = f'{n:03d}'
        frames.append(load_frame(frame_id, data_dir))
    return frames


def print_frame_summary(frame: SERVCTFrame) -> None:
    """Print calibration parameters for a frame — mirrors Assignment Q1.1–1.4."""
    print(f"Frame {frame.frame_id}  (Experiment {frame.experiment})")
    print(f"  Image size:      {frame.left_img.shape[1]} × {frame.left_img.shape[0]} px")
    print(f"  Focal length:    {frame.focal_length_px:.2f} px")
    print(f"  Principal point: cx={frame.principal_point[0]:.2f}, cy={frame.principal_point[1]:.2f}")
    print(f"  Baseline:        {frame.baseline_mm:.2f} mm")
    print(f"  GT depth range:  {frame.gt_depth_mm[frame.valid_mask].min():.1f}–"
          f"{frame.gt_depth_mm[frame.valid_mask].max():.1f} mm")
    print(f"  Valid GT pixels: {frame.valid_mask.mean()*100:.1f}%")
    print(f"\n  Q matrix:\n{frame.Q}")
