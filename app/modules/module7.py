"""Run python app/app.py; open /module-7/ for photos or /module-7/room for video.

Upload original-resolution views and the matching Module 2 calibration YAML.
Geometry and video decoding live in week_7/module7/stereo.py.
"""
import json
import shutil
import sys
import uuid
from pathlib import Path

import cv2
from flask import Blueprint, render_template, request, send_from_directory

from assignments import get

ASSIGNMENT = get('module-7')
sys.path.insert(0, ASSIGNMENT.dir)
import stereo  # noqa: E402

bp = Blueprint('module7', __name__)
NAV = [('module7.index', 'Object dimensions', '1'), ('module7.room', 'Room from video', '3')]


def run_folder(run):
    if len(run) != 32 or any(c not in '0123456789abcdef' for c in run):
        raise ValueError('Invalid measurement run.')
    folder = Path(ASSIGNMENT.uploads()) / run
    if not folder.is_dir():
        raise ValueError('Measurement run not found. Upload the views again.')
    return folder


def prepare(room):
    run = uuid.uuid4().hex
    folder = Path(ASSIGNMENT.uploads()) / run
    folder.mkdir()
    try:
        calibration = request.files.get('camera')
        if calibration and calibration.filename:
            calibration.save(folder / 'camera.yaml')
        else:
            shutil.copyfile(stereo.DEFAULT_CAMERA, folder / 'camera.yaml')
        cam = stereo.camera(folder / 'camera.yaml')
        baseline = float(request.form.get('baseline', ''))
        if not 0 < baseline < 100000:
            raise ValueError('Enter a positive baseline below 100000 mm.')
        shape = 'room' if room else request.form.get('shape', 'rectangle')
        if shape not in ('room', 'rectangle', 'circle', 'polygon'):
            raise ValueError('Invalid shape.')
        if room:
            video = request.files.get('video')
            if not video or not video.filename:
                raise ValueError('Upload a room video.')
            suffix = Path(video.filename).suffix.lower()
            if suffix not in ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'):
                raise ValueError('Use MP4, MOV, AVI, MKV or WebM video.')
            path = folder / ('video' + suffix)
            video.save(path)
            times = [float(request.form.get(key, '')) for key in ('left_time', 'right_time')]
            if times[1] <= times[0]:
                raise ValueError('The right-view time must follow the left-view time.')
            views = [stereo.frame_at(path, t) for t in times]
            path.unlink()
        else:
            times = None
            views = []
            for side in ('left', 'right'):
                upload = request.files.get(side)
                if not upload or not upload.filename:
                    raise ValueError('Upload both photos.')
                path = folder / (side + '.upload')
                upload.save(path)
                views.append(cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION))
                path.unlink()
        for side, image in zip(('left', 'right'), views):
            if image is None:
                raise ValueError('Could not decode a photo.')
            if (image.shape[1], image.shape[0]) != cam.image_size:
                raise ValueError(f'Views must match calibration resolution {cam.width} x {cam.height}. Calibrate in the same photo/video mode; do not resize or crop.')
            if not cv2.imwrite(str(folder / (side + '.png')), image):
                raise ValueError('Could not save the view.')
        meta = {'baseline_mm': baseline, 'shape': shape, 'times_s': times}
        (folder / 'meta.json').write_text(json.dumps(meta))
        return {'run': run, **meta}
    except Exception:
        shutil.rmtree(folder)
        raise


def page(room=False):
    pair, result, error = None, None, None
    if request.method == 'POST':
        try:
            if request.form.get('action') == 'measure':
                run = request.form.get('run', '')
                folder = run_folder(run)
                meta = json.loads((folder / 'meta.json').read_text())
                if (meta['shape'] == 'room') != room:
                    raise ValueError('Use the page for this measurement type.')
                pair = {'run': run, **meta}
                points = json.loads(request.form.get('points', '{}'))
                if not isinstance(points, dict):
                    raise ValueError('Invalid points.')
                result = stereo.measure(points.get('left', []), points.get('right', []),
                                        stereo.camera(folder / 'camera.yaml'), meta['baseline_mm'], meta['shape'])
                result['times_s'] = meta['times_s']
                (folder / 'result.json').write_text(json.dumps(result, indent=2))
            else:
                pair = prepare(room)
        except (ValueError, TypeError, OSError, stereo.CalibrationError, cv2.error) as exc:
            error = str(exc) if not isinstance(exc, cv2.error) else 'Could not process the calibration or media.'
    return render_template('module7/index.html', room=room, pair=pair, result=result, error=error), 400 if error else 200


@bp.route('/', methods=['GET', 'POST'])
def index():
    return page()


@bp.route('/room', methods=['GET', 'POST'])
def room():
    return page(True)


@bp.route('/output/<run>/<name>')
def output(run, name):
    if name not in ('left.png', 'right.png', 'result.json'):
        return 'Not found', 404
    try:
        return send_from_directory(run_folder(run), name)
    except ValueError:
        return 'Not found', 404
