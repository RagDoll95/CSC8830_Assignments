"""Assignment 7: simple calibrated stereo (FPCV-4-1 pp. 12-13) for objects and
two-view handheld stereo with an estimated essential matrix (FPCV-4-2) for rooms.

Run the web UI: python app/app.py, then /module-7/.
CLI: python week_7/module7/stereo.py --camera camera_params.yaml \
    --points points.json --baseline-mm 200 --shape rectangle
Shapes: rectangle (4 corners), circle (2 diameter endpoints), polygon (3+
vertices) or edge (2 endpoints of one straight distance, e.g. a room wall).
points.json: {"left": [[u,v], ...], "right": [[u,v], ...]} in matching order.
Requires the root requirements.txt. Both views must use the calibrated image
size, lens and focus, with right camera translated +B along the left X axis.
Click original distorted images; points are undistorted once.
"""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'week_2/module2/measurement'))
from geometry import Camera, CalibrationError, load_params  # noqa: E402,F401

DEFAULT_CAMERA = ROOT / 'week_2/module2/calibration/output/camera_params.yaml'
SHAPES = ('rectangle', 'circle', 'polygon', 'edge')
ROOM_LABELS = ('Width', 'Length', 'Height')
MAX_PIXEL_ERROR = 3.


def camera(path):
    cam = load_params(str(path))
    if (not np.isfinite(cam.K).all() or not np.isfinite(cam.dist).all()
            or cam.fx <= 0 or cam.fy <= 0
            or not np.allclose(cam.K[2], [0, 0, 1])
            or not np.isclose(cam.K[0, 1], 0)
            or not np.isclose(cam.K[1, 0], 0)):
        raise ValueError('Invalid pinhole calibration matrix or distortion coefficients.')
    return cam


def measure(left, right, cam, baseline_mm, shape):
    """Recover each point's Z first, then measure Euclidean 3D distances."""
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    counts = {'rectangle': 4, 'circle': 2, 'edge': 2}
    if shape not in SHAPES:
        raise ValueError('Choose rectangle, circle, polygon or edge.')
    if (a.ndim != 2 or a.shape[1:] != (2,) or b.shape != a.shape
            or len(a) < 2 or len(a) > 100 or not np.isfinite([a, b]).all()):
        raise ValueError('Mark matching finite pixel coordinates in both views (2 to 100 points).')
    if (shape in counts and len(a) != counts[shape]) or (shape == 'polygon' and len(a) < 3):
        raise ValueError('Rectangle: 4 corners; circle: 2 diameter endpoints; polygon: 3+ vertices; edge: 2 endpoints.')
    if not np.isfinite(baseline_mm) or baseline_mm <= 0:
        raise ValueError('Baseline must be a positive measured distance in millimetres.')
    for points in (a, b):
        if (points < 0).any() or (points[:, 0] >= cam.width).any() or (points[:, 1] >= cam.height).any():
            raise ValueError('A marked point is outside the calibrated image.')
    # Return normalized rays (X/Z, Y/Z); do not divide by focal length again.
    l = cv2.undistortPoints(a.reshape(-1, 1, 2), cam.K, cam.dist).reshape(-1, 2)
    r = cv2.undistortPoints(b.reshape(-1, 1, 2), cam.K, cam.dist).reshape(-1, 2)
    disparity = (l[:, 0] - r[:, 0]) * cam.fx
    vertical_error = np.abs(l[:, 1] - r[:, 1]) * cam.fy
    if not np.isfinite([l, r]).all() or (disparity <= 1).any():
        raise ValueError('Disparity must exceed 1 pixel. Check view order, matches and baseline.')
    z = baseline_mm * cam.fx / disparity
    xyz = np.column_stack((l[:, 0] * z, l[:, 1] * z, z))
    if shape in ('circle', 'edge'):
        edges, labels = [(0, 1)], ['Diameter' if shape == 'circle' else 'Distance']
    else:
        edges = [(i, (i + 1) % len(a)) for i in range(len(a))]
        labels = ['Width', 'Height', 'Opposite width', 'Opposite height'] if shape == 'rectangle' else [f'Edge {i+1}' for i in range(len(a))]
    lengths = np.array([np.linalg.norm(xyz[i] - xyz[j]) for i, j in edges])
    if (lengths <= 1e-6).any():
        raise ValueError('Dimension endpoints must be distinct.')
    warnings = []
    if (vertical_error > MAX_PIXEL_ERROR).any():
        warnings.append(f'Vertical mismatch exceeds {MAX_PIXEL_ERROR:g} pixels after undistortion; the result is an estimate. Check clicks or camera rotation.')
    if (disparity < 10).any():
        warnings.append('Small disparity: a one-pixel click error can substantially change depth.')
    return {'shape': shape, 'baseline_mm': baseline_mm,
            'left_pixels': a.tolist(), 'right_pixels': b.tolist(),
            'disparity_px': disparity.tolist(), 'vertical_error_px': vertical_error.tolist(),
            'points_mm': xyz.tolist(),
            'dimensions': [{'label': label, 'points': [i+1, j+1], 'mm': float(length)}
                           for label, (i, j), length in zip(labels, edges, lengths)],
            'warnings': warnings}


MIN_POSE_INLIERS = 30
MIN_PARALLAX_DEG = 0.5
LOW_PARALLAX_DEG = 2.


@dataclass(frozen=True)
class Pose:
    """Second view relative to the first: x_second = rotation @ x_first + translation.

    The translation has unit length; handheld video gives no metric scale, so
    distances are scaled later from a known reference length.
    """
    rotation: np.ndarray
    translation: np.ndarray
    inliers: int
    parallax_deg: float

    def to_dict(self) -> dict:
        """Return a JSON-serializable copy."""
        return {'rotation': self.rotation.tolist(), 'translation': self.translation.tolist(),
                'inliers': self.inliers, 'parallax_deg': self.parallax_deg}

    @classmethod
    def from_dict(cls, data: dict) -> 'Pose':
        """Rebuild a pose saved with to_dict."""
        return cls(np.array(data['rotation'], dtype=float), np.array(data['translation'], dtype=float),
                   int(data['inliers']), float(data['parallax_deg']))


def _normalized(pixels: np.ndarray, cam: Camera) -> np.ndarray:
    return cv2.undistortPoints(pixels.reshape(-1, 1, 2), cam.K, cam.dist).reshape(-1, 2)


def _feature_matches(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sift = cv2.SIFT_create(4000)
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(second, cv2.COLOR_BGR2GRAY), None)
    if des1 is None or des2 is None or len(kp1) < MIN_POSE_INLIERS or len(kp2) < MIN_POSE_INLIERS:
        raise ValueError('Too little texture to match the two frames. Choose frames showing more detail.')
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    a = np.array([kp1[m.queryIdx].pt for m in good], dtype=float).reshape(-1, 2)
    b = np.array([kp2[m.trainIdx].pt for m in good], dtype=float).reshape(-1, 2)
    return a, b


def _median_parallax_deg(first: np.ndarray, second: np.ndarray, rotation: np.ndarray) -> float:
    rays1 = np.column_stack((first, np.ones(len(first))))
    rays2 = np.column_stack((second, np.ones(len(second)))) @ rotation
    cosines = np.sum(rays1 * rays2, axis=1) / (np.linalg.norm(rays1, axis=1) * np.linalg.norm(rays2, axis=1))
    return float(np.degrees(np.median(np.arccos(np.clip(cosines, -1, 1)))))


def relative_pose(first: np.ndarray, second: np.ndarray, cam: Camera) -> Pose:
    """Estimate how a handheld camera moved between two frames.

    Matches SIFT features, fits the essential matrix robustly (USAC) to the
    undistorted matches and decomposes it into rotation and unit translation
    (uncalibrated-stereo pipeline with known intrinsics).

    Args:
        first: first frame, BGR, calibrated resolution.
        second: second frame, BGR, calibrated resolution.
        cam: calibration of the camera that recorded both frames.

    Returns:
        The second view's pose relative to the first.

    Raises:
        ValueError: if the frames share too few features or the camera only
            rotated, which leaves depth undetermined.
    """
    a, b = _feature_matches(first, second)
    if len(a) < MIN_POSE_INLIERS:
        raise ValueError('Too few matching features between the frames. Choose frames closer together in time.')
    pa = cv2.undistortPoints(a.reshape(-1, 1, 2), cam.K, cam.dist, P=cam.K).reshape(-1, 2)
    pb = cv2.undistortPoints(b.reshape(-1, 1, 2), cam.K, cam.dist, P=cam.K).reshape(-1, 2)
    essential, mask = cv2.findEssentialMat(pa, pb, cam.K, method=cv2.USAC_ACCURATE, prob=0.9999, threshold=1.)
    if essential is None:
        raise ValueError('Could not estimate camera motion between the frames.')
    inliers, rotation, translation, mask = cv2.recoverPose(essential[:3], pa, pb, cam.K, mask=mask)
    if inliers < MIN_POSE_INLIERS:
        raise ValueError('Could not determine how the camera moved. Choose frames that share detail and between which you stepped sideways.')
    used = mask.ravel() > 0
    parallax = _median_parallax_deg(_normalized(a[used], cam), _normalized(b[used], cam), rotation)
    if parallax < MIN_PARALLAX_DEG:
        raise ValueError('The camera turned without moving. Choose frames between which you stepped sideways.')
    return Pose(rotation, translation.ravel() / np.linalg.norm(translation), int(inliers), parallax)


def _check_pixels(a: np.ndarray, b: np.ndarray, cam: Camera, count: int) -> None:
    if a.shape != (count, 2) or b.shape != a.shape or not np.isfinite([a, b]).all():
        raise ValueError(f'Mark {count} matching points in both frames.')
    for points in (a, b):
        if (points < 0).any() or (points[:, 0] >= cam.width).any() or (points[:, 1] >= cam.height).any():
            raise ValueError('A marked point is outside the calibrated image.')


def _reprojection_error_px(xyz: np.ndarray, rays: np.ndarray, cam: Camera) -> np.ndarray:
    projected = xyz[:, :2] / xyz[:, 2:]
    return np.hypot((projected[:, 0] - rays[:, 0]) * cam.fx, (projected[:, 1] - rays[:, 1]) * cam.fy)


def measure_handheld(first: list, second: list, cam: Camera, pose: Pose, reference_mm: float) -> dict:
    """Measure one edge from two handheld frames, scaled by a known length.

    Points 1-2 are the ends of a reference of known length and points 3-4 the
    ends of the edge to measure, clicked in the same order in both frames.
    Rays are triangulated with the estimated pose, then every distance is
    scaled so that the reference has its known length.

    Args:
        first: four [u, v] clicks in the first frame (original pixels).
        second: the same four points in the second frame.
        cam: calibration of the camera.
        pose: second frame relative to the first, from relative_pose.
        reference_mm: true length of the reference, millimetres.

    Returns:
        mm (edge length), camera_move_mm, points_mm (first-camera frame),
        reprojection_error_px per view, clicked pixels and warnings.

    Raises:
        ValueError: for invalid clicks or reference, or clicks that do not
            triangulate in front of both cameras.
    """
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    _check_pixels(a, b, cam, 4)
    if not np.isfinite(reference_mm) or reference_mm <= 0:
        raise ValueError('Reference length must be a positive number of millimetres.')
    na, nb = _normalized(a, cam), _normalized(b, cam)
    projection = np.hstack((pose.rotation, pose.translation.reshape(3, 1)))
    homogeneous = cv2.triangulatePoints(np.hstack((np.eye(3), np.zeros((3, 1)))), projection, na.T, nb.T)
    xyz = (homogeneous[:3] / homogeneous[3]).T
    xyz_second = xyz @ pose.rotation.T + pose.translation
    if (xyz[:, 2] <= 0).any() or (xyz_second[:, 2] <= 0).any():
        raise ValueError('A marked point does not triangulate in front of both cameras. Check that the clicks correspond.')
    reference = np.linalg.norm(xyz[1] - xyz[0])
    if reference <= 1e-9 or np.linalg.norm(xyz[3] - xyz[2]) <= 1e-9:
        raise ValueError('Reference and edge endpoints must be distinct.')
    points_mm = xyz * (reference_mm / reference)
    errors = [_reprojection_error_px(xyz, na, cam), _reprojection_error_px(xyz_second, nb, cam)]
    warnings = []
    if max(e.max() for e in errors) > MAX_PIXEL_ERROR:
        warnings.append(f'Clicks disagree with the camera motion by more than {MAX_PIXEL_ERROR:g} pixels; the result is an estimate.')
    if pose.parallax_deg < LOW_PARALLAX_DEG:
        warnings.append('The camera moved little between the frames; depth is uncertain. Step further sideways.')
    return {'mm': float(np.linalg.norm(points_mm[3] - points_mm[2])), 'reference_mm': reference_mm,
            'camera_move_mm': float(reference_mm / reference), 'left_pixels': a.tolist(), 'right_pixels': b.tolist(),
            'points_mm': points_mm.tolist(), 'reprojection_error_px': [e.tolist() for e in errors],
            'warnings': warnings}


def room_summary(edges: list[dict]) -> list[dict]:
    """Combine separately measured room edges into width, length and height.

    Each dimension may be measured several times, from different stereo pairs
    of the same video. Repeats are averaged and their spread is reported so a
    disagreeing measurement is visible.

    Args:
        edges: measured edges, each with a 'label' from ROOM_LABELS and 'mm'.

    Returns:
        One entry per measured dimension, in ROOM_LABELS order, with the
        label, mean_mm, count and spread_mm (largest minus smallest).
    """
    summary = []
    for label in ROOM_LABELS:
        values = [edge['mm'] for edge in edges if edge['label'] == label]
        if values:
            summary.append({'label': label, 'mean_mm': float(np.mean(values)),
                            'count': len(values), 'spread_mm': float(np.ptp(values))})
    return summary


def frame_at(path, seconds):
    """Extract a frame by index, rejecting timestamps beyond the recording."""
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError('Frame times must be finite and nonnegative.')
    cap = cv2.VideoCapture(str(path))
    try:
        fps, total = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError('Cannot decode this video.')
        index = int(round(seconds * fps))
        if index >= total:
            raise ValueError('Selected time is outside the video.')
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = cap.read()
        if not ok:
            raise ValueError('Could not read the selected video frame.')
        return image
    finally:
        cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--camera', type=Path, default=DEFAULT_CAMERA)
    parser.add_argument('--points', type=Path, required=True)
    parser.add_argument('--baseline-mm', type=float, required=True)
    parser.add_argument('--shape', choices=SHAPES, required=True)
    args = parser.parse_args()
    points = json.loads(args.points.read_text())
    print(json.dumps(measure(points['left'], points['right'], camera(args.camera), args.baseline_mm, args.shape), indent=2))
