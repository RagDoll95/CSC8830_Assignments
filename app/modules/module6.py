"""Run python app/app.py and open /module-6/ for video optical flow."""
from pathlib import Path
import shutil
import sys
import uuid
import cv2
from flask import Blueprint, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from assignments import get

ASSIGNMENT = get('module-6')
if ASSIGNMENT.dir not in sys.path:
    sys.path.insert(0, ASSIGNMENT.dir)
from motion import process_video
bp = Blueprint('module6', __name__)
NAV = [('module6.index', 'Optical flow', 'A')]


@bp.route('/', methods=['GET', 'POST'])
def index():
    result, error = None, None
    if request.method == 'POST':
        run = uuid.uuid4().hex
        folder = Path(ASSIGNMENT.uploads()) / run
        folder.mkdir()
        try:
            upload = request.files.get('video')
            if not upload or not upload.filename:
                raise ValueError('Choose a video file.')
            suffix = Path(secure_filename(upload.filename)).suffix.lower()
            if suffix not in {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}:
                raise ValueError('Use a video file such as MP4 or AVI.')
            source = folder / ('input' + suffix)
            upload.save(source)
            video = process_video(source, folder, float(request.form.get('start', 0)))
            source.unlink()
            result = {'run': run, 'video': video}
        except (ValueError, cv2.error) as exc:
            shutil.rmtree(folder)
            error = str(exc) if isinstance(exc, ValueError) else 'Could not process this video.'
    return render_template('module6/index.html', result=result, error=error), 400 if error else 200


@bp.route('/output/<run>/<name>')
def output(run, name):
    if len(run) != 32 or any(c not in '0123456789abcdef' for c in run):
        return 'Run not found', 404
    return send_from_directory(Path(ASSIGNMENT.uploads()) / run, name)
