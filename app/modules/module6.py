"""Run python app/app.py and open /module-6/.

Part A (/module-6/): 30 seconds of sparse optical flow for a bundled sample or
an uploaded video. Bundled samples are also checked against hand-measured
pixel locations in their first two frames (validate.py).

Part B (/module-6/structure): four-view planar structure from motion with the
Module 2 phone calibration (sfm.py). Runs on the bundled checkerboard photos,
on four uploaded checkerboard photos, or on four photos whose corresponding
points are clicked in the browser.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import uuid
from pathlib import Path

import cv2
from flask import Blueprint, render_template, request, send_from_directory
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from assignments import get

ASSIGNMENT = get('module-6')
if ASSIGNMENT.dir not in sys.path:
    sys.path.insert(0, ASSIGNMENT.dir)
from motion import process_video  # noqa: E402
import results  # noqa: E402
import sfm  # noqa: E402
from validate import flow_summary  # noqa: E402

bp = Blueprint('module6', __name__)
NAV = [('module6.index', 'Optical flow and tracking', 'A'),
       ('module6.structure', 'Structure from motion', 'B')]
VIDEO_TYPES = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
PHOTO_TYPES = {'.jpg', '.jpeg', '.png'}
CACHED_RUNS = set(results.SAMPLES) | {results.STRUCTURE}


def new_run() -> tuple[str, Path]:
    run = uuid.uuid4().hex
    folder = Path(ASSIGNMENT.uploads()) / run
    folder.mkdir()
    return run, folder


def read_rows(path: Path) -> list[dict]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def video_name(folder: Path) -> str:
    return 'flow.mp4' if (folder / 'flow.mp4').is_file() else 'flow.avi'


def flow_upload(upload: FileStorage) -> dict:
    suffix = Path(secure_filename(upload.filename)).suffix.lower()
    if suffix not in VIDEO_TYPES:
        raise ValueError('Use a video file such as MP4 or AVI.')
    run, folder = new_run()
    try:
        source = folder / ('input' + suffix)
        upload.save(source)
        video = process_video(source, folder, float(request.form.get('start', 0)))
        source.unlink()
        return {'run': run, 'video': video, 'summary': flow_summary(folder), 'validation': None}
    except (ValueError, cv2.error):
        shutil.rmtree(folder)
        raise


def flow_sample(sample: str) -> dict:
    if sample not in results.SAMPLES:
        raise ValueError('Choose a sample video.')
    folder = results.sample_results(sample)
    return {'run': sample, 'video': video_name(folder),
            'summary': json.loads((folder / 'flow_summary.json').read_text()),
            'validation': read_rows(folder / 'validation.csv')}


@bp.route('/', methods=['GET', 'POST'])
def index():
    result, error = None, None
    if request.method == 'POST':
        upload = request.files.get('video')
        try:
            if upload and upload.filename:
                result = flow_upload(upload)
            else:
                result = flow_sample(request.form.get('sample', 'penguins'))
        except (ValueError, cv2.error) as exc:
            error = str(exc) if isinstance(exc, ValueError) else 'Could not process this video.'
    return render_template('module6/index.html', result=result, error=error), 400 if error else 200


def save_photos(files: list[FileStorage], folder: Path) -> list[Path]:
    photos = [f for f in files if f and f.filename]
    if len(photos) != 4:
        raise ValueError('Upload exactly four photos, one per viewpoint.')
    paths = []
    for k, photo in enumerate(photos, start=1):
        suffix = Path(secure_filename(photo.filename)).suffix.lower()
        if suffix not in PHOTO_TYPES:
            raise ValueError('Use JPEG or PNG photos (export HEIC photos as JPEG first).')
        path = folder / f'view{k}{suffix}'
        photo.save(path)
        paths.append(path)
    return paths


def clicked_points(folder: Path) -> Path:
    try:
        data = {'points': json.loads(request.form.get('points') or '[]'),
                'known': [int(request.form.get('known_i', 1)) - 1,
                          int(request.form.get('known_j', 2)) - 1,
                          float(request.form.get('known_mm', 0))]}
    except ValueError as exc:
        raise ValueError('Points and the known distance must be numbers.') from exc
    path = folder / 'points.json'
    path.write_text(json.dumps(data))
    return path


def structure_upload() -> dict:
    run, folder = new_run()
    try:
        images = save_photos(request.files.getlist('photos'), folder)
        if request.form.get('mode') == 'checkerboard':
            return {'run': run, 'summary': sfm.run(folder, images)}
        return {'run': run, 'summary': sfm.run(folder, images, clicked_points(folder))}
    except (ValueError, cv2.error):
        shutil.rmtree(folder)
        raise


def structure_bundled() -> dict:
    folder = results.structure_results()
    return {'run': results.STRUCTURE,
            'summary': json.loads((folder / 'sfm_calculations.json').read_text())}


@bp.route('/structure', methods=['GET', 'POST'])
def structure():
    result, error = None, None
    if request.method == 'POST':
        try:
            if request.form.get('source', 'bundled') == 'bundled':
                result = structure_bundled()
            else:
                result = structure_upload()
        except (ValueError, cv2.error) as exc:
            error = str(exc) if isinstance(exc, ValueError) else 'Could not process these photos.'
    return render_template('module6/structure.html', result=result, error=error), 400 if error else 200


@bp.route('/output/<run>/<name>')
def output(run, name):
    if run in CACHED_RUNS and results.is_complete(results.OUTPUT / run):
        return send_from_directory(results.OUTPUT / run, name)
    if len(run) != 32 or any(c not in '0123456789abcdef' for c in run):
        return 'Run not found', 404
    return send_from_directory(Path(ASSIGNMENT.uploads()) / run, name)
