"""Run python app/app.py; open /module-7/ for photos or /module-7/room for video.

Upload original-resolution views and the matching Module 2 calibration YAML.
Room mode takes one handheld phone video and measures each room edge
separately: for every edge, pick two frames filmed a step apart while facing
it, the app estimates the camera motion between them, then click a reference
of known length and the edge's two endpoints in both frames. Geometry and
video decoding live in week_7/module7/stereo.py.
"""
from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from flask import Blueprint, render_template, request, send_from_directory

from assignments import get

ASSIGNMENT = get('module-7')
sys.path.insert(0, ASSIGNMENT.dir)
import stereo  # noqa: E402

bp = Blueprint('module7', __name__)
NAV = [('module7.index', 'Object dimensions', '1'), ('module7.room', 'Room from video', '3')]
VIDEO_SUFFIXES = ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v')
HANDLED = (ValueError, TypeError, KeyError, OSError, stereo.CalibrationError, cv2.error)


def run_folder(run: str) -> Path:
    """Return the upload folder of an existing measurement run.

    Raises:
        ValueError: if the run id is malformed or the folder does not exist.
    """
    if len(run) != 32 or any(c not in '0123456789abcdef' for c in run):
        raise ValueError('Invalid measurement run.')
    folder = Path(ASSIGNMENT.uploads()) / run
    if not folder.is_dir():
        raise ValueError('Measurement run not found. Upload the views again.')
    return folder


def _length(value: str, name: str) -> float:
    length = float(value)
    if not 0 < length < 100000:
        raise ValueError(f'Enter a positive {name} below 100000 mm.')
    return length


def _save_calibration(folder: Path) -> stereo.Camera:
    calibration = request.files.get('camera')
    if calibration and calibration.filename:
        calibration.save(folder / 'camera.yaml')
    else:
        shutil.copyfile(stereo.DEFAULT_CAMERA, folder / 'camera.yaml')
    return stereo.camera(folder / 'camera.yaml')


def _save_views(folder: Path, views: list[np.ndarray | None], cam: stereo.Camera) -> None:
    for side, image in zip(('left', 'right'), views):
        if image is None:
            raise ValueError('Could not decode a view.')
        if (image.shape[1], image.shape[0]) != cam.image_size:
            raise ValueError(f'{side.capitalize()} view is {image.shape[1]} x {image.shape[0]} pixels after orientation correction. Views must match calibration resolution {cam.width} x {cam.height}. Upload the original full-resolution file from the calibrated capture mode; do not resize or crop.')
        if not cv2.imwrite(str(folder / f'{side}.png'), image):
            raise ValueError('Could not save the view.')


def _read_meta(folder: Path) -> dict:
    return json.loads((folder / 'meta.json').read_text())


def _write_meta(folder: Path, meta: dict) -> None:
    (folder / 'meta.json').write_text(json.dumps(meta))


def _points() -> dict:
    points = json.loads(request.form.get('points', '{}'))
    if not isinstance(points, dict):
        raise ValueError('Invalid points.')
    return points


def _new_run(build: Callable[[Path], dict]) -> dict:
    run = uuid.uuid4().hex
    folder = Path(ASSIGNMENT.uploads()) / run
    folder.mkdir()
    try:
        meta = build(folder)
    except Exception:
        shutil.rmtree(folder)
        raise
    _write_meta(folder, meta)
    return {'run': run, **meta}


def _prepare_photos(folder: Path) -> dict:
    cam = _save_calibration(folder)
    baseline = _length(request.form.get('baseline', ''), 'baseline')
    shape = request.form.get('shape', 'rectangle')
    if shape not in ('rectangle', 'circle', 'polygon'):
        raise ValueError('Invalid shape.')
    views = []
    for side in ('left', 'right'):
        upload = request.files.get(side)
        if not upload or not upload.filename:
            raise ValueError('Upload both photos.')
        path = folder / f'{side}.upload'
        upload.save(path)
        views.append(cv2.imread(str(path), cv2.IMREAD_COLOR))
        path.unlink()
    _save_views(folder, views, cam)
    return {'baseline_mm': baseline, 'shape': shape}


def _prepare_video(folder: Path) -> dict:
    cam = _save_calibration(folder)
    video = request.files.get('video')
    if not video or not video.filename:
        raise ValueError('Upload a room video.')
    suffix = Path(video.filename).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise ValueError('Use MP4, MOV, AVI, MKV or WebM video.')
    video.save(folder / f'video{suffix}')
    first = stereo.frame_at(folder / f'video{suffix}', 0)
    if (first.shape[1], first.shape[0]) != cam.image_size:
        raise ValueError(f'Video frames must match calibration resolution {cam.width} x {cam.height}. Calibrate checkerboard frames from this video mode.')
    return {'shape': 'room', 'video': f'video{suffix}', 'pair': None, 'edges': []}


def _photo_page() -> tuple[dict | None, dict | None]:
    if request.form.get('action') != 'measure':
        return _new_run(_prepare_photos), None
    run = request.form.get('run', '')
    folder = run_folder(run)
    meta = _read_meta(folder)
    if meta['shape'] == 'room':
        raise ValueError('Use the room page for this measurement.')
    points = _points()
    result = stereo.measure(points.get('left', []), points.get('right', []),
                            stereo.camera(folder / 'camera.yaml'), meta['baseline_mm'], meta['shape'])
    (folder / 'result.json').write_text(json.dumps(result, indent=2))
    return {'run': run, **meta}, result


def _load_pair(folder: Path, meta: dict) -> None:
    label = request.form.get('label', '')
    if label not in stereo.ROOM_LABELS:
        raise ValueError('Choose width, length or height.')
    times = [float(request.form.get(key, '')) for key in ('left_time', 'right_time')]
    if not np.isfinite(times).all() or times[0] == times[1]:
        raise ValueError('Choose two different frame times.')
    cam = stereo.camera(folder / 'camera.yaml')
    frames = [stereo.frame_at(folder / meta['video'], t) for t in times]
    _save_views(folder, frames, cam)
    meta['pair'] = {'label': label, 'times_s': times, 'pose': stereo.relative_pose(*frames, cam).to_dict()}


def _measure_edge(folder: Path, meta: dict) -> None:
    pair = meta['pair']
    if pair is None:
        raise ValueError('Load a stereo pair before marking points.')
    points = _points()
    reference_mm = _length(request.form.get('reference_mm', ''), 'reference length')
    result = stereo.measure_handheld(points.get('left', []), points.get('right', []),
                                     stereo.camera(folder / 'camera.yaml'), stereo.Pose.from_dict(pair['pose']),
                                     reference_mm)
    meta['edges'].append({'label': pair['label'], 'times_s': pair['times_s'], **result})
    meta['pair'] = None


def _remove_edge(meta: dict) -> None:
    index = int(request.form.get('index', ''))
    if not 0 <= index < len(meta['edges']):
        raise ValueError('No such measured edge.')
    meta['edges'].pop(index)


def _write_room_result(folder: Path, meta: dict) -> None:
    result = {'edges': meta['edges'], 'summary': stereo.room_summary(meta['edges'])}
    (folder / 'result.json').write_text(json.dumps(result, indent=2))


def _room_page() -> dict:
    action = request.form.get('action', 'upload')
    if action == 'upload':
        return _new_run(_prepare_video)
    run = request.form.get('run', '')
    folder = run_folder(run)
    meta = _read_meta(folder)
    if meta['shape'] != 'room':
        raise ValueError('Use the object page for this measurement.')
    steps = {'pair': lambda: _load_pair(folder, meta),
             'measure': lambda: _measure_edge(folder, meta),
             'remove': lambda: _remove_edge(meta)}
    if action not in steps:
        raise ValueError('Unknown action.')
    steps[action]()
    _write_meta(folder, meta)
    _write_room_result(folder, meta)
    return {'run': run, **meta}


def _render(room: bool, pair: dict | None, result: dict | None, error: str | None):
    summary = stereo.room_summary(pair['edges']) if room and pair else []
    return render_template('module7/index.html', room=room, pair=pair, result=result,
                           summary=summary, labels=stereo.ROOM_LABELS, error=error), 400 if error else 200


def _error_message(exc: Exception) -> str:
    if isinstance(exc, cv2.error):
        return 'Could not process the calibration or media.'
    return str(exc)


@bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method != 'POST':
        return _render(False, None, None, None)
    try:
        pair, result = _photo_page()
    except HANDLED as exc:
        return _render(False, None, None, _error_message(exc))
    return _render(False, pair, result, None)


@bp.route('/room', methods=['GET', 'POST'])
def room():
    if request.method != 'POST':
        return _render(True, None, None, None)
    run = request.form.get('run', '')
    try:
        state = _room_page()
    except HANDLED as exc:
        return _render(True, _existing_room(run), None, _error_message(exc))
    return _render(True, state, None, None)


def _existing_room(run: str) -> dict | None:
    try:
        meta = _read_meta(run_folder(run))
    except HANDLED:
        return None
    return {'run': run, **meta} if meta.get('shape') == 'room' else None


@bp.route('/output/<run>/<name>')
def output(run: str, name: str):
    if name not in ('left.png', 'right.png', 'result.json'):
        return 'Not found', 404
    try:
        return send_from_directory(run_folder(run), name)
    except ValueError:
        return 'Not found', 404
