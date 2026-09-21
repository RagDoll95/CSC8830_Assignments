"""Module 3 web page: compare spatial and Fourier-domain image blurring."""

import os
import sys
import uuid

import cv2
from flask import Blueprint, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from assignments import get

ASSIGNMENT = get("module-3")
MODULE = ASSIGNMENT.dir
if MODULE not in sys.path:
    sys.path.insert(0, MODULE)

from image_blur import display_image, process  # noqa: E402

OUTPUT = os.path.join(MODULE, "output")
ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

bp = Blueprint("module3", __name__)
NAV = [("module3.index", "Blur and compare", "")]


def _save(name, image):
    os.makedirs(OUTPUT, exist_ok=True)
    cv2.imwrite(os.path.join(OUTPUT, name), image)


@bp.route("/", methods=["GET", "POST"])
def index():
    context = {"result": None, "error": None, "values": {
        "kind": "gaussian", "size": 9, "sigma": 2.0,
    }}
    if request.method == "GET":
        return render_template("module3/index.html", **context)

    upload = request.files.get("image")
    try:
        kind = request.form.get("kind", "gaussian")
        size = int(request.form.get("size", "9"))
        sigma = float(request.form.get("sigma", "2.0"))
        context["values"] = {"kind": kind, "size": size, "sigma": sigma}
        if not upload or not upload.filename:
            raise ValueError("Choose an image to process.")
        extension = os.path.splitext(secure_filename(upload.filename))[1].lower()
        if extension not in ALLOWED:
            raise ValueError("Use a JPG, PNG, BMP, or TIFF image.")
        data = upload.read()
        image = cv2.imdecode(__import__("numpy").frombuffer(data, dtype="uint8"),
                             cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("The uploaded file could not be read as an image.")

        spatial, fourier, difference, metrics = process(image, kind, size, sigma)
        run = uuid.uuid4().hex[:12]
        names = {
            "original": run + "-original.png",
            "spatial": run + "-spatial.png",
            "fourier": run + "-fourier.png",
            "difference": run + "-difference.png",
        }
        _save(names["original"], image)
        _save(names["spatial"], display_image(spatial))
        _save(names["fourier"], display_image(fourier))
        _save(names["difference"], difference)
        context["result"] = {"names": names, "metrics": metrics}
    except (TypeError, ValueError) as exc:
        context["error"] = str(exc)
        return render_template("module3/index.html", **context), 400
    return render_template("module3/index.html", **context)


@bp.route("/output/<path:name>")
def output(name):
    return send_from_directory(OUTPUT, secure_filename(name))
