"""Generate clearly labelled synthetic stereo inputs (not real-world evidence).

Run: python week_7/module7/demo.py --out /tmp/module7-demo
Upload camera.yaml, baseline 200 mm. For object use left.png and right.png;
for room use room.avi at times 0 and 2 seconds. points.json contains exact clicks.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

K = np.array([[800., 0, 640], [0, 800, 360], [0, 0, 1]])
B = 200.
OBJECT = np.array([[-500., -300, 3000], [500, -300, 3000], [500, 300, 3000], [-500, 300, 3000]])
# Three perpendicular edges, rotated so all four endpoints are in both views.
origin = np.array([-1000., 700, 4000])
ROOM = np.array([origin, origin + [2400, 0, 1800], origin + [-1200, 0, 1600], origin + [0, -2400, 0]])


def project(xyz, right=False):
    shifted = xyz - np.array([B if right else 0, 0, 0])
    pixels = shifted @ K.T
    return pixels[:, :2] / pixels[:, 2:]


def generate(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(out / 'camera.yaml'), cv2.FILE_STORAGE_WRITE)
    fs.write('camera_matrix', K)
    fs.write('distortion_coefficients', np.zeros((1, 5)))
    fs.write('image_width', 1280)
    fs.write('image_height', 720)
    fs.release()
    all_points = {}
    for kind, xyz in [('object', OBJECT), ('room', ROOM)]:
        images, points = [], {}
        for side in ('left', 'right'):
            p = project(xyz, side == 'right')
            points[side] = p.tolist()
            image = np.full((720, 1280, 3), 245, np.uint8)
            cv2.putText(image, f'SYNTHETIC {kind.upper()} - {side} view', (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
            edges = [(0, 1), (0, 2), (0, 3)] if kind == 'room' else [(0, 1), (1, 2), (2, 3), (3, 0)]
            for i, j in edges:
                cv2.line(image, tuple(np.rint(p[i]).astype(int)), tuple(np.rint(p[j]).astype(int)), (120, 80, 20), 3)
            for i, point in enumerate(p):
                x, y = np.rint(point).astype(int)
                cv2.drawMarker(image, (x, y), (0, 0, 220), cv2.MARKER_CROSS, 15, 2)
                cv2.putText(image, str(i+1), (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            images.append(image)
            if kind == 'object':
                cv2.imwrite(str(out / (side + '.png')), image)
        all_points[kind] = points
        if kind == 'room':
            writer = cv2.VideoWriter(str(out / 'room.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 10, (1280, 720))
            if not writer.isOpened():
                raise ValueError('MJPG video encoder unavailable.')
            for index in range(40):
                writer.write(images[0 if index < 20 else 1])
            writer.release()
    (out / 'points.json').write_text(json.dumps(all_points, indent=2))
    (out / 'object_points.json').write_text(json.dumps(all_points['object'], indent=2))
    return all_points


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    generate(parser.parse_args().out)
