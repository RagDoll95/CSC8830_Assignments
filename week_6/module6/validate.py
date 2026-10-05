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


def cube_derivatives(first, second):
    """I_x, I_y, I_t from the 2x2x2 cube of pixels (FPCV-4-3, slide 18).

    Each is the mean of the four cube values on the far face minus the mean on
    the near face; element [y, x] uses pixels x..x+1, y..y+1 of both frames.
    """
    cube = first + second
    ix = (cube[:-1, 1:] + cube[1:, 1:] - cube[:-1, :-1] - cube[1:, :-1]) / 4
    iy = (cube[1:, :-1] + cube[1:, 1:] - cube[:-1, :-1] - cube[:-1, 1:]) / 4
    change = second - first
    it = (change[:-1, :-1] + change[:-1, 1:] + change[1:, :-1] + change[1:, 1:]) / 4
    return ix, iy, it


def ssd_match(first, second, x, y, half=10, search=10):
    """Integer location in the second frame minimizing SSD (FPCV-5-1, slide 25)."""
    template = first[y-half:y+half+1, x-half:x+half+1]
    top, left = max(0, y-half-search), max(0, x-half-search)
    region = second[top:y+half+search+1, left:x+half+search+1]
    scores = cv2.matchTemplate(region, template, cv2.TM_SQDIFF)
    dy, dx = np.unravel_index(np.argmin(scores), scores.shape)
    return np.array([left + dx + half, top + dy + half], float)


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
    ix, iy, it = cube_derivatives(first, second)
    calculations = []
    with (folder / 'validation.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['point', 'x0', 'y0', 'measured_x1', 'measured_y1',
                         'tracked_x1', 'tracked_y1', 'error_px', 'valid',
                         'ssd_x1', 'ssd_y1', 'ssd_error_px'])
        for row, p, actual, q, good in zip(rows, points, measured, tracked, status[:, 0]):
            x, y = p.astype(int)
            patch = np.s_[y-10:y+11, x-10:x+11]
            a = np.column_stack([ix[patch].ravel(), iy[patch].ravel()])
            b = -it[patch].ravel()
            delta = np.linalg.lstsq(a, b, rcond=None)[0]
            calculations.append({'point': row['point'], 'AtA': (a.T@a).tolist(),
                                 'AtB': (a.T@b).tolist(), 'linear_step': delta.tolist(),
                                 'eigenvalues': np.linalg.eigvalsh(a.T@a).tolist()})
            error = float(np.linalg.norm(q-actual)) if good else None
            ssd = ssd_match(gray[0], gray[1], x, y)
            writer.writerow([row['point'], *p, *actual, *q, error, int(good),
                             *ssd, float(np.linalg.norm(ssd - actual))])
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


def flow_summary(folder):
    """Separate camera motion from object motion in the 30-second flow field.

    The per-frame median vector estimates the dominant (camera) motion; a
    vector more than 1 pixel from it is moving differently, through its own
    motion or through depth parallax.
    Writes flow_summary.png and flow_summary.json into folder.
    """
    folder = Path(folder)
    data = np.genfromtxt(folder / 'points.csv', delimiter=',', names=True)
    frame = data['frame'].astype(int)
    flow = np.column_stack([data['dx'], data['dy']])
    frames = np.unique(frame)
    median = np.array([np.median(flow[frame == k], axis=0) for k in frames])
    residual = np.linalg.norm(flow - median[np.searchsorted(frames, frame)], axis=1)
    magnitude = np.linalg.norm(flow, axis=1)
    summary = {'vectors': int(len(flow)), 'frames': int(len(frames)),
               'median_speed_px_per_frame': float(np.median(magnitude)),
               'median_camera_motion_px_per_frame': float(np.median(np.linalg.norm(median, axis=1))),
               'static_fraction': float(np.mean(magnitude < 0.5)),
               'differs_from_dominant_fraction': float(np.mean(residual > 1)),
               'frame0_camera_motion': median[0].tolist()}
    (folder / 'flow_summary.json').write_text(json.dumps(summary, indent=2))
    pair = cv2.cvtColor(cv2.imread(str(folder / 'frame0.png')), cv2.COLOR_BGR2RGB)
    first = frame == frames[0]
    moving = residual[first] > 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), layout='constrained')
    axes[0].imshow(pair)
    starts = np.column_stack([data['x0'], data['y0']])[first]
    axes[0].quiver(*starts[~moving].T, *(flow[first][~moving] * 4).T, color='cyan', angles='xy',
                   scale_units='xy', scale=1, width=.003, label='follows dominant motion')
    axes[0].quiver(*starts[moving].T, *(flow[first][moving] * 4).T, color='magenta', angles='xy',
                   scale_units='xy', scale=1, width=.003, label='differs by >1 px (object motion or parallax)')
    axes[0].set_title('Frames 0 to 1, vectors x4'); axes[0].axis('off')
    axes[0].legend(loc='lower left', fontsize=8)
    seconds = frames / (len(frames) + 1) * 30
    axes[1].plot(seconds, median[:, 0], label='camera motion dx (median)')
    axes[1].plot(seconds, median[:, 1], label='camera motion dy (median)')
    spread = [np.percentile(residual[frame == k], 90) for k in frames]
    axes[1].plot(seconds, spread, color='gray', lw=.8, label='90th pct. deviation from median')
    axes[1].set_xlabel('time in sample (s)'); axes[1].set_ylabel('pixels / frame')
    axes[1].set_ylim(-15, 25); axes[1].grid(True); axes[1].legend(fontsize=8)
    axes[1].set_title('Dominant motion over 30 seconds')
    fig.savefig(folder / 'flow_summary.png', dpi=150)
    plt.close(fig)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sample', choices=['penguins', 'marble_race'])
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    from motion import process_video
    process_video(Path(__file__).with_name(args.sample+'.mp4'), args.out)
    validate_pair(args.out, args.sample)
    print(json.dumps(flow_summary(args.out), indent=2))
