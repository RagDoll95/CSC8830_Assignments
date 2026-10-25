"""Generate clearly labelled synthetic stereo inputs (not real-world evidence).

Run: python week_7/module7/demo.py --out /tmp/module7-demo
Upload camera.yaml, baseline 200 mm. For object use left.png and right.png.
For room use room.avi. The camera stands inside a 3000 x 4000 x 2400 mm room:
stops at 0.5 s and 1.5 s face the far wall (width and height edges), stops at
2.5 s and 3.5 s face the side wall (length edge). points.json has exact clicks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

K = np.array([[800., 0, 640], [0, 800, 360], [0, 0, 1]])
SIZE = (1280, 720)
B = 200.
OBJECT = np.array([[-500., -300, 3000], [500, -300, 3000], [500, 300, 3000], [-500, 300, 3000]])
ROOM_SIZE = {'Width': 3000., 'Length': 4000., 'Height': 2400.}
# World frame: X along the width, Y down (floor at +1200, ceiling at -1200), Z along the length.
ROOM_EDGES = {
    'Width': np.array([[0., 1200, 4000], [3000, 1200, 4000]]),
    'Height': np.array([[0., 1200, 4000], [0, -1200, 4000]]),
    'Length': np.array([[3000., 1200, 0], [3000, 1200, 4000]]),
}
# Each view: rows of the rotation are the camera's x (right), y (down) and z (forward) axes in the world.
FACING_FAR_WALL = (np.eye(3), np.array([1500., 0, 500]))
FACING_SIDE_WALL = (np.array([[0., 0, -1], [0, 1, 0], [1, 0, 0]]), np.array([200., 0, 2100]))
ROOM_PAIRS = [
    {'label': 'Width', 'view': FACING_FAR_WALL, 'times_s': [0.5, 1.5]},
    {'label': 'Height', 'view': FACING_FAR_WALL, 'times_s': [0.5, 1.5]},
    {'label': 'Length', 'view': FACING_SIDE_WALL, 'times_s': [2.5, 3.5]},
]
FPS = 10


def project(xyz: np.ndarray, rotation: np.ndarray = np.eye(3), center: np.ndarray = np.zeros(3),
            right: bool = False) -> np.ndarray:
    """Project world points into the left camera, or into the right camera B mm along its x axis."""
    origin = center + (B * rotation[0] if right else 0)
    camera = (xyz - origin) @ rotation.T
    if (camera[:, 2] <= 0).any():
        raise ValueError('A synthetic point is behind the camera.')
    pixels = camera @ K.T
    return pixels[:, :2] / pixels[:, 2:]


def _canvas(title: str) -> np.ndarray:
    image = np.full((SIZE[1], SIZE[0], 3), 245, np.uint8)
    cv2.putText(image, title, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
    return image


def _draw(image: np.ndarray, pixels: np.ndarray, edges: list[tuple[int, int]], first_label: int = 1) -> None:
    for i, j in edges:
        cv2.line(image, tuple(np.rint(pixels[i]).astype(int)), tuple(np.rint(pixels[j]).astype(int)), (120, 80, 20), 3)
    for i, point in enumerate(pixels):
        x, y = np.rint(point).astype(int)
        cv2.drawMarker(image, (x, y), (0, 0, 220), cv2.MARKER_CROSS, 15, 2)
        cv2.putText(image, str(i + first_label), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def _write_camera(out: Path) -> None:
    fs = cv2.FileStorage(str(out / 'camera.yaml'), cv2.FILE_STORAGE_WRITE)
    fs.write('camera_matrix', K)
    fs.write('distortion_coefficients', np.zeros((1, 5)))
    fs.write('image_width', SIZE[0])
    fs.write('image_height', SIZE[1])
    fs.release()


def _object_views(out: Path) -> dict:
    points = {}
    for side in ('left', 'right'):
        pixels = project(OBJECT, right=side == 'right')
        points[side] = pixels.tolist()
        image = _canvas(f'SYNTHETIC OBJECT - {side} view')
        _draw(image, pixels, [(0, 1), (1, 2), (2, 3), (3, 0)])
        cv2.imwrite(str(out / f'{side}.png'), image)
    return points


def _room_frame(view: tuple[np.ndarray, np.ndarray], right: bool) -> np.ndarray:
    side = 'right' if right else 'left'
    wall = 'far wall' if view is FACING_FAR_WALL else 'side wall'
    image = _canvas(f'SYNTHETIC ROOM - facing {wall} - {side} stop')
    labels = [pair['label'] for pair in ROOM_PAIRS if pair['view'] is view]
    for label in labels:
        _draw(image, project(ROOM_EDGES[label], *view, right=right), [(0, 1)])
    return image


def _room_video(out: Path) -> list[dict]:
    writer = cv2.VideoWriter(str(out / 'room.avi'), cv2.VideoWriter_fourcc(*'MJPG'), FPS, SIZE)
    if not writer.isOpened():
        raise ValueError('MJPG video encoder unavailable.')
    for view in (FACING_FAR_WALL, FACING_SIDE_WALL):
        for right in (False, True):
            frame = _room_frame(view, right)
            for _ in range(FPS):
                writer.write(frame)
    writer.release()
    return [{'label': pair['label'], 'times_s': pair['times_s'],
             'left': project(ROOM_EDGES[pair['label']], *pair['view']).tolist(),
             'right': project(ROOM_EDGES[pair['label']], *pair['view'], right=True).tolist()}
            for pair in ROOM_PAIRS]


def generate(out: Path | str) -> dict:
    """Write camera.yaml, object photos, room.avi and the exact click coordinates.

    Args:
        out: output folder, created if missing.

    Returns:
        {'object': {'left', 'right'}, 'room': [{'label', 'times_s', 'left', 'right'}, ...]}.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _write_camera(out)
    all_points = {'object': _object_views(out), 'room': _room_video(out)}
    (out / 'points.json').write_text(json.dumps(all_points, indent=2))
    (out / 'object_points.json').write_text(json.dumps(all_points['object'], indent=2))
    return all_points


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    generate(parser.parse_args().out)
