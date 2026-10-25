"""Assignment 7: simple calibrated stereo, FPCV-4-1 pp. 12-13.

Run the web UI: python app/app.py, then /module-7/.
CLI: python week_7/module7/stereo.py --camera camera_params.yaml \
    --points points.json --baseline-mm 200 --shape rectangle
points.json: {"left": [[u,v], ...], "right": [[u,v], ...]} in matching order.
Requires the root requirements.txt. Both views must use the calibrated image
size, lens and focus, with right camera translated +B along the left X axis,
no rotation. Click original distorted images; points are undistorted once.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'week_2/module2/measurement'))
from geometry import load_params, CalibrationError  # noqa: E402

DEFAULT_CAMERA = ROOT / 'week_2/module2/calibration/output/camera_params.yaml'


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
    counts = {'rectangle': 4, 'circle': 2, 'room': 4}
    if shape not in (*counts, 'polygon'):
        raise ValueError('Choose rectangle, circle, polygon or room.')
    if (a.ndim != 2 or a.shape[1:] != (2,) or b.shape != a.shape
            or len(a) < 2 or len(a) > 100 or not np.isfinite([a, b]).all()):
        raise ValueError('Mark matching finite pixel coordinates in both views (2 to 100 points).')
    if (shape in counts and len(a) != counts[shape]) or (shape == 'polygon' and len(a) < 3):
        raise ValueError('Rectangle: 4 corners; circle: 2 diameter endpoints; polygon: 3+ vertices; room: 4 corners.')
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
    if (vertical_error > 3).any():
        raise ValueError('Vertical mismatch exceeds 3 pixels after undistortion. Check clicks or recapture without camera rotation.')
    z = baseline_mm * cam.fx / disparity
    xyz = np.column_stack((l[:, 0] * z, l[:, 1] * z, z))
    if shape == 'room':
        edges, labels = [(0, 1), (0, 2), (0, 3)], ['Width', 'Length', 'Height']
    elif shape == 'circle':
        edges, labels = [(0, 1)], ['Diameter']
    else:
        edges = [(i, (i + 1) % len(a)) for i in range(len(a))]
        labels = ['Width', 'Height', 'Opposite width', 'Opposite height'] if shape == 'rectangle' else [f'Edge {i+1}' for i in range(len(a))]
    lengths = np.array([np.linalg.norm(xyz[i] - xyz[j]) for i, j in edges])
    if (lengths <= 1e-6).any():
        raise ValueError('Dimension endpoints must be distinct.')
    warnings = []
    if (disparity < 10).any():
        warnings.append('Small disparity: a one-pixel click error can substantially change depth.')
    if shape == 'room':
        directions = (xyz[1:] - xyz[0]) / lengths[:, None]
        if np.max(np.abs((directions @ directions.T) - np.eye(3))) > 0.2:
            warnings.append('The three room edges are not close to perpendicular. Check the common corner and corresponding endpoints.')
    return {'shape': shape, 'baseline_mm': baseline_mm,
            'left_pixels': a.tolist(), 'right_pixels': b.tolist(),
            'disparity_px': disparity.tolist(), 'vertical_error_px': vertical_error.tolist(),
            'points_mm': xyz.tolist(),
            'dimensions': [{'label': label, 'points': [i+1, j+1], 'mm': float(length)}
                           for label, (i, j), length in zip(labels, edges, lengths)],
            'warnings': warnings}


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
    parser.add_argument('--shape', choices=['rectangle', 'circle', 'polygon', 'room'], required=True)
    args = parser.parse_args()
    points = json.loads(args.points.read_text())
    print(json.dumps(measure(points['left'], points['right'], camera(args.camera), args.baseline_mm, args.shape), indent=2))
