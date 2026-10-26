"""Generate clearly labelled synthetic stereo inputs (not real-world evidence).

Run: python week_7/module7/demo.py --out /tmp/module7-demo
Upload camera.yaml. For object use left.png and right.png, baseline 200 mm.
For room use room.avi: a wobbling handheld camera in a 3000 x 4000 x 2400 mm
room walks past the far wall (0-2 s; width and height edges, frames 0.3 s and
1.7 s) then the side wall (2-4 s; length edge, frames 2.3 s and 3.7 s). The
top edge of the top poster on each wall is the 800 mm scale reference.
points.json has exact clicks: reference ends first, then edge ends.
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
POSTER_MM = (800., 600.)
# Posters on walls, floor and furniture give the frames texture at several depths; (center, right axis, down axis).
# The first poster of each walk is the scale reference.
FAR_POSTERS = [((1500., -300, 4000), (1, 0, 0), (0, 1, 0)), ((600., 300, 4000), (1, 0, 0), (0, 1, 0)),
               ((2400., 400, 4000), (1, 0, 0), (0, 1, 0)), ((1000., 1200, 3400), (1, 0, 0), (0, 0, -1)),
               ((2100., 1200, 3600), (1, 0, 0), (0, 0, -1)), ((2300., 700, 2200), (1, 0, 0), (0, 1, 0)),
               ((800., -100, 2600), (1, 0, 0), (0, 1, 0))]
SIDE_POSTERS = [((3000., -300, 2000), (0, 0, -1), (0, 1, 0)), ((3000., 300, 3200), (0, 0, -1), (0, 1, 0)),
                ((3000., 400, 900), (0, 0, -1), (0, 1, 0)), ((1700., 700, 2900), (0, 0, -1), (0, 1, 0)),
                ((1900., 0, 1100), (0, 0, -1), (0, 1, 0)), ((2200., 1000, 2000), (0, 0, -1), (0, 1, 0))]
# Handheld walks: start/end camera centre and look-at target per wall, filmed over 2 s each.
WALKS = [{'posters': FAR_POSTERS, 'centers': ((1500., 0, 500), (1850, 40, 420)),
          'targets': ((1500., 0, 4000), (1350, 60, 4000))},
         {'posters': SIDE_POSTERS, 'centers': ((200., 0, 2050), (180, 50, 1750)),
          'targets': ((3000., 0, 2000), (3000, 40, 1800))}]
ROOM_PAIRS = [
    {'label': 'Width', 'walk': 0, 'times_s': [0.3, 1.7]},
    {'label': 'Height', 'walk': 0, 'times_s': [0.3, 1.7]},
    {'label': 'Length', 'walk': 1, 'times_s': [2.3, 3.7]},
]
SEGMENT_S = 2
FPS = 10


def project(xyz: np.ndarray, rotation: np.ndarray = np.eye(3), center: np.ndarray = np.zeros(3),
            right: bool = False) -> np.ndarray:
    """Project world points into a camera, or into its stereo partner B mm along its x axis.

    Rows of rotation are the camera's x (right), y (down) and z (forward) axes in the world.
    """
    origin = center + (B * rotation[0] if right else 0)
    camera = (np.asarray(xyz) - origin) @ rotation.T
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


def handheld_view(seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the (rotation, centre) of the wobbling handheld camera at a video time."""
    frame = int(round(seconds * FPS))
    walk = WALKS[frame // (SEGMENT_S * FPS)]
    s = frame % (SEGMENT_S * FPS) / (SEGMENT_S * FPS - 1)
    wobble = np.array([np.sin(9 * s), np.cos(7 * s), 0])
    center = (1 - s) * np.array(walk['centers'][0]) + s * np.array(walk['centers'][1]) + 15 * wobble
    target = (1 - s) * np.array(walk['targets'][0]) + s * np.array(walk['targets'][1]) + 60 * wobble[::-1]
    forward = (target - center) / np.linalg.norm(target - center)
    right = np.cross([0, 1, 0], forward)
    right /= np.linalg.norm(right)
    roll = 0.02 * np.sin(5 * s)
    rotation = np.array([right, np.cross(forward, right), forward])
    spin = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    return spin @ rotation, center


def poster_corners(poster: tuple) -> np.ndarray:
    """Top-left, top-right, bottom-right and bottom-left corners of a poster in the world."""
    center, right, down = (np.array(v, dtype=float) for v in poster)
    half_w, half_h = right * POSTER_MM[0] / 2, down * POSTER_MM[1] / 2
    return np.array([center - half_w - half_h, center + half_w - half_h, center + half_w + half_h, center - half_w + half_h])


def _texture(seed: int) -> np.ndarray:
    blocks = np.random.default_rng(seed).integers(0, 256, (12, 16, 3), dtype=np.uint8)
    return cv2.resize(blocks, (320, 240), interpolation=cv2.INTER_NEAREST)


def _draw_poster(image: np.ndarray, poster: tuple, seed: int, view: tuple[np.ndarray, np.ndarray]) -> None:
    texture = _texture(seed)
    h, w = texture.shape[:2]
    source = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    homography = cv2.getPerspectiveTransform(source, np.float32(project(poster_corners(poster), *view)))
    cv2.warpPerspective(texture, homography, SIZE, dst=image, borderMode=cv2.BORDER_TRANSPARENT)


def _room_frame(seconds: float) -> np.ndarray:
    walk = int(round(seconds * FPS)) // (SEGMENT_S * FPS)
    return render_room(handheld_view(seconds), walk, f'SYNTHETIC ROOM - handheld - {seconds:.1f} s')


def render_room(view: tuple[np.ndarray, np.ndarray], walk: int, title: str) -> np.ndarray:
    """Draw one walk's posters and room edges as seen from a (rotation, centre) view."""
    image = _canvas(title)
    for index, poster in enumerate(WALKS[walk]['posters']):
        _draw_poster(image, poster, 10 * walk + index, view)
    for label in [pair['label'] for pair in ROOM_PAIRS if pair['walk'] == walk]:
        _draw(image, project(ROOM_EDGES[label], *view), [(0, 1)])
    return image


def _room_clicks(pair: dict) -> dict:
    reference = poster_corners(WALKS[pair['walk']]['posters'][0])[:2]
    points = np.vstack((reference, ROOM_EDGES[pair['label']]))
    clicks = {side: project(points, *handheld_view(t)) for side, t in zip(('left', 'right'), pair['times_s'])}
    for pixels in clicks.values():
        if (pixels < 0).any() or (pixels >= SIZE).any():
            raise ValueError('A synthetic click is outside the frame.')
    return {'label': pair['label'], 'times_s': pair['times_s'], 'reference_mm': POSTER_MM[0],
            'left': clicks['left'].tolist(), 'right': clicks['right'].tolist()}


def _room_video(out: Path) -> list[dict]:
    writer = cv2.VideoWriter(str(out / 'room.avi'), cv2.VideoWriter_fourcc(*'MJPG'), FPS, SIZE)
    if not writer.isOpened():
        raise ValueError('MJPG video encoder unavailable.')
    for frame in range(len(WALKS) * SEGMENT_S * FPS):
        writer.write(_room_frame(frame / FPS))
    writer.release()
    return [_room_clicks(pair) for pair in ROOM_PAIRS]


def generate(out: Path | str) -> dict:
    """Write camera.yaml, object photos, room.avi and the exact click coordinates.

    Args:
        out: output folder, created if missing.

    Returns:
        {'object': {'left', 'right'},
         'room': [{'label', 'times_s', 'reference_mm', 'left', 'right'}, ...]}.
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
