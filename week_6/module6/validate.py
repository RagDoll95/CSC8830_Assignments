"""Run: python week_6/module6/validate.py penguins --out output
Compare the first two sample frames with visually selected integer pixel locations.
Requires the root requirements.txt. Measurements are approximate, not exact ground truth.
"""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def validate_pair(folder, sample):
    folder = Path(folder)
    with Path(__file__).with_name('measurements.csv').open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['sample'] == sample]
    frames = [cv2.imread(str(folder / f'frame{i}.png')) for i in range(2)]
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    points = np.array([[float(r['x0']), float(r['y0'])] for r in rows], np.float32)
    measured = np.array([[float(r['x1']), float(r['y1'])] for r in rows])
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        *gray, points.reshape(-1, 1, 2), None, winSize=(21, 21), maxLevel=3)
    tracked = tracked[:, 0]
    # One full-resolution linear step: the normal equations derived in the slides.
    first, second = [f.astype(float) for f in gray]
    ix = cv2.Sobel(first, cv2.CV_64F, 1, 0, ksize=3, scale=1/8)
    iy = cv2.Sobel(first, cv2.CV_64F, 0, 1, ksize=3, scale=1/8)
    calculations = []
    with (folder / 'validation.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['point', 'x0', 'y0', 'measured_x1', 'measured_y1',
                         'tracked_x1', 'tracked_y1', 'error_px', 'valid'])
        for row, p, actual, q, good in zip(rows, points, measured, tracked, status[:, 0]):
            x, y = p.astype(int)
            patch = np.s_[y-10:y+11, x-10:x+11]
            a = np.column_stack([ix[patch].ravel(), iy[patch].ravel()])
            b = -(second-first)[patch].ravel()
            delta = np.linalg.lstsq(a, b, rcond=None)[0]
            calculations.append({'point': row['point'], 'AtA': (a.T@a).tolist(),
                                 'Atb': (a.T@b).tolist(), 'linear_step': delta.tolist()})
            error = float(np.linalg.norm(q-actual)) if good else None
            writer.writerow([row['point'], *p, *actual, *q, error, int(good)])
    (folder / 'tracking_calculations.json').write_text(json.dumps(calculations, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for i, ax in enumerate(axes):
        ax.imshow(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
        coords = points if i == 0 else measured
        ax.scatter(*coords.T, s=90, facecolors='none', edgecolors='lime', label='Visual selection')
        for j, (x, y) in enumerate(coords):
            ax.annotate(str(j+1), (x, y), xytext=(5, 8), textcoords='offset points', color='lime')
        if i == 1:
            valid = status[:, 0].astype(bool)
            ax.scatter(*tracked[valid].T, marker='+', color='red', label='Lucas-Kanade')
        ax.set_xlim(max(0, points[:, 0].min()-30), points[:, 0].max()+30)
        ax.set_ylim(points[:, 1].max()+30, max(0, points[:, 1].min()-20))
        ax.set_title(f'Frame {i}'); ax.set_xlabel('x (pixels)'); ax.set_ylabel('y (pixels)')
        ax.legend(loc='lower right', fontsize=8)
    fig.savefig(folder / 'validation.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sample', choices=['penguins', 'marble_race'])
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    from motion import process_video
    process_video(Path(__file__).with_name(args.sample+'.mp4'), args.out)
    validate_pair(args.out, args.sample)
