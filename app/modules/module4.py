"""Module 4 upload page. Run from repo root: python app/app.py.

Visit /module-4/; uploads use the same fixed OpenCV pipeline as the CLI.
"""
import os
import sys
import uuid

import cv2
import numpy as np
from flask import Blueprint, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from assignments import get

ASSIGNMENT = get('module-4')
if ASSIGNMENT.dir not in sys.path:
    sys.path.insert(0, ASSIGNMENT.dir)
from boundaries import detect_boundaries  # noqa: E402

bp = Blueprint('module4', __name__)
NAV = [('module4.index', 'RGB and thermal boundaries', '')]
ALLOWED = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


@bp.route('/', methods=['GET', 'POST'])
def index():
    result, error = None, None
    if request.method == 'POST':
        try:
            upload = request.files.get('image')
            if not upload or not upload.filename:
                raise ValueError('Choose an RGB or thermal image.')
            if os.path.splitext(secure_filename(upload.filename))[1].lower() not in ALLOWED:
                raise ValueError('Use a JPG, PNG, BMP, or TIFF image.')
            data = upload.read()
            if not data:
                raise ValueError('The uploaded image is empty.')
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            stages, count = detect_boundaries(image)
            run = uuid.uuid4().hex
            names = {}
            for key, stage in stages.items():
                names[key] = run + '-' + key + '.png'
                if not cv2.imwrite(os.path.join(ASSIGNMENT.uploads(), names[key]), stage):
                    raise OSError('Could not save the result image.')
            result = {'names': names, 'count': count}
        except (ValueError, cv2.error) as exc:
            error = str(exc) if isinstance(exc, ValueError) else 'This image could not be processed.'
    return render_template('module4/index.html', result=result, error=error), 400 if error else 200


@bp.route('/output/<name>')
def output(name):
    return send_from_directory(ASSIGNMENT.uploads(), secure_filename(name))
