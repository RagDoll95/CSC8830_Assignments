"""Four-view orthographic structure from motion (Tomasi-Kanade) of a flat object.

Run from the repository root (requires the root requirements.txt):
    python week_6/module6/sfm.py --out week_6/module6/output/structure
    python week_6/module6/sfm.py --images a.jpg b.jpg c.jpg d.jpg \
        --points points.json --out output/structure

With no --images, four checkerboard photos from Module 2 are used and their
9x6 inner corners (24.9 mm squares) are found automatically. The known grid is
used only to fix the metric scale from one distance and to grade the result.

points.json for your own photos:
    {"points": [[[x, y], ...], x4 views, same order in every view],
     "known": [i, j, millimetres between points i and j]}
The points are listed in boundary order. Lens distortion and the intrinsics are
removed with week_2/module2/calibration/output/camera_params.yaml, so photos
must come from that phone at the same zoom. Take the four views from about the
same distance: the orthographic model assumes one magnification for all frames.

Method (FPCV-4-4, Tomasi-Kanade factorization):
    1. centroid-subtract every frame (centering trick) and stack W (2F x N),
    2. W = M S has rank <= 2 for a plane, so keep the two largest singular values,
       M = U1 S1^(1/2) Q, S = Q^-1 S1^(1/2) V1^T,
    3. find Q with Newton's method from the orthonormality of every camera's
       i_f, j_f; a plane leaves their out-of-plane parts as extra unknowns,
    4. scale S by one known distance and draw the boundary.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CALIBRATION = REPO_ROOT / 'week_2' / 'module2' / 'calibration' / 'output'
CAMERA_FILE = CALIBRATION / 'camera_params.yaml'
CHECKERBOARD_PHOTOS = [CALIBRATION / 'debug' / f'IMG_{n}.jpeg' for n in (3930, 3917, 3928, 3938)]
PATTERN = (9, 6)
SQUARE_MM = 24.9
NEWTON_STARTS = 40
NEWTON_ITERATIONS = 200


@dataclass
class Camera:
    """Intrinsics of the calibrated phone camera."""

    matrix: np.ndarray
    distortion: np.ndarray
    size: tuple[int, int]


@dataclass
class Views:
    """Four photos and their corresponding pixel points."""

    images: list[Path]
    points: np.ndarray
    boundary: list[int]
    known: tuple[int, int, float]
    truth_mm: np.ndarray | None = None


@dataclass
class Factorization:
    """Every quantity of the factorization; image units are K^-1-normalized."""

    observation: np.ndarray
    centroids: np.ndarray
    singular_values: np.ndarray
    motion_hat: np.ndarray
    q: np.ndarray
    motion: np.ndarray
    out_of_plane: np.ndarray
    orthonormality_residuals: np.ndarray
    shape: np.ndarray
    mm_per_unit: float
    predicted_px: np.ndarray
    reprojection_px: np.ndarray

    @property
    def shape_mm(self) -> np.ndarray:
        """Recovered in-plane coordinates of every point, (N, 2) millimetres."""
        return (self.shape * self.mm_per_unit).T


def load_camera(path: Path = CAMERA_FILE) -> Camera:
    """Read the OpenCV calibration file written by Module 2."""
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        return Camera(storage.getNode('camera_matrix').mat(),
                      storage.getNode('distortion_coefficients').mat(),
                      (int(storage.getNode('image_width').real()),
                       int(storage.getNode('image_height').real())))
    finally:
        storage.release()


def camera_for(images: list[Path], camera: Camera) -> Camera:
    """Return the camera rescaled to the photos' resolution.

    Raises:
        ValueError: if a photo is unreadable or has a different aspect ratio.
    """
    sizes = set()
    for path in images:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f'Could not read image {path.name}.')
        sizes.add((image.shape[1], image.shape[0]))
    if len(sizes) != 1:
        raise ValueError('All four photos must have the same resolution.')
    width, height = sizes.pop()
    scale = width / camera.size[0]
    if abs(height / camera.size[1] - scale) > 1e-3:
        raise ValueError(f'Photos are {width}x{height}; the calibration is '
                         f'{camera.size[0]}x{camera.size[1]}. Use the same phone, '
                         'lens, orientation and aspect ratio as the calibration.')
    matrix = camera.matrix.copy()
    matrix[:2] *= scale
    return Camera(matrix, camera.distortion, (width, height))


def grid_boundary(cols: int, rows: int) -> list[int]:
    """Indices of a cols x rows grid's outer corners, walked around the edge."""
    top = list(range(cols))
    right = [r * cols + cols - 1 for r in range(1, rows)]
    bottom = [(rows - 1) * cols + c for c in range(cols - 2, -1, -1)]
    left = [r * cols for r in range(rows - 2, 0, -1)]
    return top + right + bottom + left


def checkerboard_views(images: list[Path] | None = None,
                       pattern: tuple[int, int] = PATTERN,
                       square_mm: float = SQUARE_MM) -> Views:
    """Detect the checkerboard's inner corners in four photos.

    Raises:
        ValueError: if the pattern is not found in every photo.
    """
    images = images or CHECKERBOARD_PHOTOS
    points = []
    for path in images:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        found, corners = cv2.findChessboardCornersSB(gray, pattern) if gray is not None else (False, None)
        if not found:
            raise ValueError(f'No {pattern[0]}x{pattern[1]} checkerboard found in {path.name}.')
        points.append(corners.reshape(-1, 2))
    cols, rows = pattern
    truth = np.zeros((cols * rows, 3))
    truth[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return Views(list(images), np.array(points, float), grid_boundary(cols, rows),
                 (0, cols - 1, (cols - 1) * square_mm), truth)


def clicked_views(images: list[Path], data: dict) -> Views:
    """Views from points marked by hand, listed in boundary order.

    Raises:
        ValueError: if the points or known distance are malformed.
    """
    points = np.asarray(data.get('points'), float)
    if points.ndim != 3 or points.shape[0] != 4 or points.shape[2] != 2 or points.shape[1] < 4:
        raise ValueError('Mark the same 4 or more points, in the same order, in all four photos.')
    if not np.isfinite(points).all():
        raise ValueError('Point coordinates must be finite numbers.')
    i, j, mm = data.get('known', (0, 1, 0))
    i, j, mm = int(i), int(j), float(mm)
    if not (0 <= i < points.shape[1] and 0 <= j < points.shape[1] and i != j and mm > 0):
        raise ValueError('Give a positive real distance between two different marked points.')
    return Views(images, points, list(range(points.shape[1])), (i, j, mm))


def normalized_points(points: np.ndarray, camera: Camera) -> np.ndarray:
    """Undistorted image coordinates with K removed, shape (F, N, 2)."""
    return np.array([cv2.undistortPoints(view.reshape(-1, 1, 2).astype(np.float32),
                                         camera.matrix, camera.distortion)[:, 0]
                     for view in points], dtype=float)


def observation_matrix(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centering trick: W (2F x N) of centroid-subtracted u rows, then v rows."""
    centroids = points.mean(axis=1)
    centered = points - centroids[:, None, :]
    return np.vstack([centered[:, :, 0], centered[:, :, 1]]), centroids


def orthonormality(params: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Residuals |i|^2-1, |j|^2-1, i.j for every frame, and their Jacobian.

    params = (Q row-major, alpha_1..F, beta_1..F). The camera axes are
    i_f = (a_f^T Q, alpha_f) and j_f = (b_f^T Q, beta_f) in a frame whose third
    axis is the plane normal; alpha, beta are invisible in W for a flat object.
    """
    frames = len(a)
    q = params[:4].reshape(2, 2)
    alpha, beta = params[4:4 + frames], params[4 + frames:]
    i, j = a @ q, b @ q
    residuals = np.concatenate([np.sum(i * i, axis=1) + alpha ** 2 - 1,
                                np.sum(j * j, axis=1) + beta ** 2 - 1,
                                np.sum(i * j, axis=1) + alpha * beta])
    d_q = np.vstack([2 * np.einsum('fk,fl->fkl', a, i).reshape(frames, 4),
                     2 * np.einsum('fk,fl->fkl', b, j).reshape(frames, 4),
                     (np.einsum('fk,fl->fkl', a, j) + np.einsum('fk,fl->fkl', b, i)).reshape(frames, 4)])
    eye = np.eye(frames)
    zero = np.zeros((frames, frames))
    d_alpha = np.vstack([2 * alpha * eye, zero, beta * eye])
    d_beta = np.vstack([zero, 2 * beta * eye, alpha * eye])
    return residuals, np.hstack([d_q, d_alpha, d_beta])


def newton(params: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Damped Gauss-Newton iterations on the orthonormality equations."""
    for _ in range(NEWTON_ITERATIONS):
        residuals, jacobian = orthonormality(params, a, b)
        normal = jacobian.T @ jacobian
        step = np.linalg.solve(normal + 1e-9 * np.trace(normal) * np.eye(len(params)),
                               -jacobian.T @ residuals)
        params = params + step
        if np.linalg.norm(step) < 1e-13:
            break
    return params


def solve_q(motion_hat: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Q and out-of-plane parts that make every camera's i_f, j_f orthonormal.

    The 3F equations are quadratic, so Newton's method is started from several
    deterministic random guesses and the best least-squares solution is kept.
    Returns Q (2x2), [alpha_f, beta_f] (F x 2) and the final residuals.
    """
    frames = len(motion_hat) // 2
    a, b = motion_hat[:frames], motion_hat[frames:]
    rng = np.random.default_rng(seed)
    scale = 1 / np.sqrt(np.mean(np.sum(motion_hat ** 2, axis=1)))
    best, best_cost = None, np.inf
    for _ in range(NEWTON_STARTS):
        start = np.concatenate([rng.normal(size=4) * scale, rng.uniform(-1, 1, 2 * frames)])
        params = newton(start, a, b)
        cost = np.linalg.norm(orthonormality(params, a, b)[0])
        if cost < best_cost:
            best, best_cost = params, cost
    residuals = orthonormality(best, a, b)[0]
    return best[:4].reshape(2, 2), best[4:].reshape(2, frames).T, residuals


def front_facing(q: np.ndarray, motion_hat: np.ndarray) -> np.ndarray:
    """Q with the in-plane axes ordered so the cameras see the plane's front.

    A plane's shape is otherwise only known up to a mirror image. With
    k_f = i_f x j_f, the normal component of every camera axis must be positive.
    """
    frames = len(motion_hat) // 2
    i, j = motion_hat[:frames] @ q, motion_hat[frames:] @ q
    normal_component = i[:, 0] * j[:, 1] - i[:, 1] * j[:, 0]
    return q if np.median(normal_component) > 0 else q @ np.diag([1.0, -1.0])


def along_first_edge(q: np.ndarray, shape_hat: np.ndarray, boundary: list[int]) -> np.ndarray:
    """Q times the in-plane rotation that puts the first boundary edge on +x.

    Q is only defined up to such a rotation; fixing it changes no image.
    """
    edge = np.linalg.solve(q, shape_hat[:, boundary[1]] - shape_hat[:, boundary[0]])
    angle = np.arctan2(edge[1], edge[0])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return q @ rotation


def reproject(points: np.ndarray, camera: Camera) -> np.ndarray:
    """Normalized coordinates (F, N, 2) back to distorted pixels."""
    rays = np.concatenate([points, np.ones(points.shape[:2] + (1,))], axis=2)
    return np.array([cv2.projectPoints(view, np.zeros(3), np.zeros(3), camera.matrix,
                                       camera.distortion)[0][:, 0] for view in rays])


def reconstruct(views: Views, camera: Camera) -> Factorization:
    """Tomasi-Kanade factorization of the four views, rank 2 for a plane.

    Raises:
        ValueError: if the points are degenerate.
    """
    w, centroids = observation_matrix(normalized_points(views.points, camera))
    u, singular, vt = np.linalg.svd(w, full_matrices=False)
    if singular[1] < 1e-9 * singular[0]:
        raise ValueError('The marked points must not all lie on one line.')
    motion_hat = u[:, :2] * np.sqrt(singular[:2])
    shape_hat = np.sqrt(singular[:2])[:, None] * vt[:2]
    q, out_of_plane, residuals = solve_q(motion_hat)
    q = along_first_edge(front_facing(q, motion_hat), shape_hat, views.boundary)
    motion, shape = motion_hat @ q, np.linalg.solve(q, shape_hat)
    i, j, mm = views.known
    mm_per_unit = mm / np.linalg.norm(shape[:, i] - shape[:, j])
    frames = len(centroids)
    predicted = (motion @ shape).reshape(2, frames, -1).transpose(1, 2, 0) + centroids[:, None, :]
    predicted_px = reproject(predicted, camera)
    errors = np.linalg.norm(predicted_px - views.points, axis=2)
    return Factorization(w, centroids, singular, motion_hat, q, motion, out_of_plane,
                         residuals, shape, float(mm_per_unit), predicted_px, errors)


def camera_axes(result: Factorization) -> list[dict]:
    """Each camera's i_f, j_f, k_f = i_f x j_f and tilt from the plane normal.

    The axes are expressed in (in-plane x, in-plane y, normal). For a plane only
    the product alpha_f * beta_f is fixed, so the out-of-plane signs are chosen
    with alpha_f >= 0: the tilt angle is recovered, not which way it leans.
    """
    frames = len(result.centroids)
    rows = []
    for f in range(frames):
        alpha, beta = result.out_of_plane[f]
        sign = 1.0 if alpha >= 0 else -1.0
        i = np.append(result.motion[f], sign * alpha)
        j = np.append(result.motion[frames + f], sign * beta)
        k = np.cross(i, j)
        tilt = np.degrees(np.arccos(np.clip(abs(k[2]) / np.linalg.norm(k), 0, 1)))
        rows.append({'view': f + 1, 'i': i.tolist(), 'j': j.tolist(), 'k': k.tolist(),
                     'tilt_deg': float(tilt)})
    return rows


def calibration_poses(views: Views, camera: Camera) -> list[dict]:
    """Camera positions measured with the known board (PnP), for reference only."""
    rows = []
    center_of_board = views.truth_mm.mean(axis=0)
    for f, pixels in enumerate(views.points, start=1):
        _, rvec, tvec = cv2.solvePnP(views.truth_mm, pixels, camera.matrix, camera.distortion)
        rotation = cv2.Rodrigues(rvec)[0]
        center = -rotation.T @ tvec.ravel()
        offset = center - center_of_board
        depths = (rotation @ views.truth_mm.T + tvec)[2]
        rows.append({'view': f, 'center_mm': center.tolist(),
                     'distance_mm': float(np.linalg.norm(offset)),
                     'tilt_deg': float(np.degrees(np.arccos(abs(offset[2]) / np.linalg.norm(offset)))),
                     'depth_range_percent': float(100 * np.ptp(depths) / depths.mean())})
    return rows


def rigid_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares rotation and translation (Kabsch) of source onto target."""
    s, t = source - source.mean(axis=0), target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(s.T @ t)
    flip = np.eye(source.shape[1])
    flip[-1, -1] = np.sign(np.linalg.det(vt.T @ u.T))
    return s @ (vt.T @ flip @ u.T).T + target.mean(axis=0)


def boundary_lengths(plane: np.ndarray, boundary: list[int]) -> list[float]:
    """Lengths of the straight sides between boundary corners where it turns."""
    loop = plane[boundary]
    d = np.roll(loop, -1, axis=0) - loop
    sine = (np.roll(d, 1, axis=0)[:, 0] * d[:, 1] - np.roll(d, 1, axis=0)[:, 1] * d[:, 0]) / (
        np.linalg.norm(d, axis=1) * np.linalg.norm(np.roll(d, 1, axis=0), axis=1))
    turns = np.flatnonzero(np.abs(sine) > np.sin(np.radians(20))).tolist()
    return [float(np.linalg.norm(loop[b] - loop[a])) for a, b in zip(turns, turns[1:] + turns[:1])]


def ground_truth_check(views: Views, result: Factorization, cameras: list[dict],
                       poses: list[dict]) -> dict:
    """Compare the structure with the known grid and tilts with PnP."""
    truth = views.truth_mm[:, :2]
    errors = np.linalg.norm(rigid_fit(truth, result.shape_mm) - result.shape_mm, axis=1)
    return {'point_rms_mm': float(np.sqrt(np.mean(errors ** 2))),
            'point_max_mm': float(errors.max()),
            'true_side_lengths_mm': boundary_lengths(truth, views.boundary),
            'tilt_error_deg': [c['tilt_deg'] - p['tilt_deg'] for c, p in zip(cameras, poses)]}


def plot_views(views: Views, result: Factorization, cameras: list[dict], path: Path) -> None:
    """Save the four photos with marked points, reprojections and boundary."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 5), layout='constrained')
    for k, ax in enumerate(axes):
        image = cv2.imread(str(views.images[k]))
        factor = 800 / image.shape[0]
        small = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        ax.imshow(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        loop = views.points[k][views.boundary + views.boundary[:1]] * factor
        ax.plot(*loop.T, '-', color='yellow', lw=1.5)
        ax.scatter(*(views.points[k] * factor).T, s=14, facecolors='none', edgecolors='lime',
                   label='Measured')
        ax.scatter(*(result.predicted_px[k] * factor).T, s=10, marker='+', color='red',
                   label='Reprojected (MS)')
        ax.set_title(f"View {k + 1}: {views.images[k].name}\nrecovered tilt {cameras[k]['tilt_deg']:.0f}°",
                     fontsize=9)
        ax.axis('off')
    axes[0].legend(loc='lower left', fontsize=8)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_structure(views: Views, result: Factorization, path: Path) -> None:
    """Save the singular values of W and the recovered boundary."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained',
                                   gridspec_kw={'width_ratios': [1, 2]})
    ax1.bar(range(1, len(result.singular_values) + 1), result.singular_values, color='tab:blue')
    ax1.set_yscale('log'); ax1.set_xlabel('index'); ax1.set_ylabel('singular value of W')
    ax1.set_title('Singular values of W: two dominate (plane)', fontsize=10)
    shape = result.shape_mm
    if views.truth_mm is not None:
        truth = rigid_fit(views.truth_mm[:, :2], shape)
        ax2.plot(*truth[views.boundary + views.boundary[:1]].T, '--', color='gray', label='True grid')
    ax2.scatter(*shape.T, s=8, color='tab:blue')
    ax2.plot(*shape[views.boundary + views.boundary[:1]].T, color='tab:blue', label='Recovered boundary')
    ax2.set_aspect('equal'); ax2.grid(True); ax2.invert_yaxis(); ax2.legend(fontsize=8)
    ax2.set_xlabel('x (mm, in plane)'); ax2.set_ylabel('y (mm, in plane)')
    ax2.set_title('Structure S (scaled by one known edge)', fontsize=10)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def calculations(views: Views, camera: Camera, result: Factorization, cameras: list[dict]) -> dict:
    """Every intermediate quantity, for the report and downloads."""
    summary = {
        'images': [p.name for p in views.images],
        'camera_matrix': camera.matrix.tolist(),
        'distortion': camera.distortion.ravel().tolist(),
        'image_size': list(camera.size),
        'known_distance': list(views.known),
        'pixels': views.points.tolist(),
        'boundary_order': views.boundary,
        'centroids_normalized': result.centroids.tolist(),
        'W': result.observation.tolist(),
        'singular_values': result.singular_values.tolist(),
        'M_hat': result.motion_hat.tolist(),
        'Q': result.q.tolist(),
        'QQt': (result.q @ result.q.T).tolist(),
        'M': result.motion.tolist(),
        'out_of_plane_alpha_beta': result.out_of_plane.tolist(),
        'orthonormality_residuals': result.orthonormality_residuals.tolist(),
        'cameras': cameras,
        'S_normalized': result.shape.tolist(),
        'mm_per_unit': result.mm_per_unit,
        'S_mm': result.shape_mm.tolist(),
        'boundary_side_lengths_mm': boundary_lengths(result.shape_mm, views.boundary),
        'reprojection_mean_px': result.reprojection_px.mean(axis=1).tolist(),
        'reprojection_max_px': result.reprojection_px.max(axis=1).tolist(),
    }
    if views.truth_mm is not None:
        poses = calibration_poses(views, camera)
        summary['calibration_poses'] = poses
        summary['ground_truth'] = ground_truth_check(views, result, cameras, poses)
    return summary


def run(output: Path, images: list[Path] | None = None, points_file: Path | None = None) -> dict:
    """Reconstruct four views and write figures, CSV and JSON into output.

    Raises:
        ValueError: for unreadable photos or inconsistent points.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if images and points_file:
        views = clicked_views(images, json.loads(Path(points_file).read_text()))
    else:
        views = checkerboard_views(images)
    camera = camera_for(views.images, load_camera())
    result = reconstruct(views, camera)
    cameras = camera_axes(result)
    summary = calculations(views, camera, result, cameras)
    (output / 'sfm_calculations.json').write_text(json.dumps(summary, indent=2))
    np.savetxt(output / 'boundary.csv',
               np.column_stack([views.boundary, result.shape_mm[views.boundary]]),
               delimiter=',', header='point,x_mm,y_mm', comments='', fmt=['%d', '%.3f', '%.3f'])
    plot_views(views, result, cameras, output / 'views.png')
    plot_structure(views, result, output / 'reconstruction.png')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--images', nargs=4, type=Path)
    parser.add_argument('--points', type=Path, help='JSON of marked points (omit for a checkerboard)')
    args = parser.parse_args()
    report = run(args.out, args.images, args.points)
    for key in ('singular_values', 'orthonormality_residuals', 'cameras', 'boundary_side_lengths_mm',
                'reprojection_mean_px', 'calibration_poses', 'ground_truth'):
        print(key, json.dumps(report.get(key)))
