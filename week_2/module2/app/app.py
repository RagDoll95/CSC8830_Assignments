"""
Part E -- web application. All four parts of the assignment reachable from one page.

HOW TO RUN
    # from module2/
    pip install -r requirements.txt
    python app/app.py
    # then open http://127.0.0.1:5000

    # options
    python app/app.py --port 8000 --host 0.0.0.0 --debug

PAGES
    /              overview + navigation
    /calibration   Step 1: upload board images, fit, download camera_params.yaml
    /measurement   Step 2: upload an image, enter Z, click two points, measure
    /validation    Step 3: load the CSV, see the table, statistics and plots
    /theory        Part D: the two-camera derivation

DESIGN RULE
    No geometry lives in this file. Every number comes from measurement/geometry.py
    and calibration/calibrate.py -- the same code the CLI uses -- so the web app
    and the CLI cannot disagree. tools/verify_pipeline.py asserts that they do not.

THE ONE REAL GOTCHA (plan section 25)
    Canvas clicks arrive in DISPLAY coordinates. A 2016-px-wide photo shown in
    an 800-px canvas means every click must be multiplied by 2016/800 = 2.52
    before it reaches the geometry, or every measurement is wrong by that
    factor. The browser therefore posts the natural image dimensions alongside
    each click, and scale_click() below does the conversion server-side, on the
    full-resolution image, because K was calibrated at full resolution.
"""

import argparse
import csv
import io
import os
import shutil
import sys
import uuid

import cv2
import numpy as np
from flask import (Flask, abort, jsonify, redirect, render_template, request,
                   send_file, send_from_directory, url_for)
from werkzeug.utils import secure_filename

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "measurement"))
sys.path.insert(0, os.path.join(ROOT, "calibration"))
sys.path.insert(0, os.path.join(ROOT, "validation"))

from geometry import (CalibrationError, annotate, check_resolution,  # noqa: E402
                      load_params, measure, measure_uncertainty)
import calibrate as calib  # noqa: E402
import analyze  # noqa: E402

UPLOAD_ROOT = os.path.join(HERE, "uploads")
DEFAULT_PARAMS = os.path.join(ROOT, "calibration", "output", "camera_params.yaml")
DEFAULT_CSV = os.path.join(ROOT, "validation", "data", "measurements.csv")
VALIDATION_OUTPUT = os.path.join(ROOT, "validation", "output")
SAMPLE_MEAS_DIR = os.path.join(ROOT, "validation", "data", "images")
CALIB_DEBUG_DIR = os.path.join(ROOT, "calibration", "output", "debug")

# Phone photos are large; cap the request and reject non-images.
MAX_CONTENT_MB = 200
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024


# --------------------------------------------------------------------------
# The section-25 fix. Kept module-level and tiny so verify_pipeline.py can
# import and test it directly against the CLI.
# --------------------------------------------------------------------------
def scale_click(x_disp, y_disp, disp_w, disp_h, nat_w, nat_h):
    """Canvas/display coordinates -> full-resolution image coordinates.

    K is in pixels at the calibration resolution, so the geometry must only
    ever see full-resolution coordinates. Returns (u, v) as floats.
    """
    if not disp_w or not disp_h:
        raise ValueError("display dimensions must be non-zero")
    return (float(x_disp) * float(nat_w) / float(disp_w),
            float(y_disp) * float(nat_h) / float(disp_h))


def allowed_image(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_IMAGE_EXT


def session_dir(token=None, create=True):
    token = token or uuid.uuid4().hex[:12]
    if not all(c in "0123456789abcdef" for c in token):
        abort(400, "bad token")
    path = os.path.join(UPLOAD_ROOT, token)
    if create:
        os.makedirs(path, exist_ok=True)
    return token, path


def active_params_path():
    """The most recent web-fitted calibration, else the CLI's output."""
    marker = os.path.join(UPLOAD_ROOT, "active_params.txt")
    if os.path.exists(marker):
        with open(marker) as fh:
            candidate = fh.read().strip()
        if candidate and os.path.exists(candidate):
            return candidate
    return DEFAULT_PARAMS


def set_active_params(path):
    os.makedirs(UPLOAD_ROOT, exist_ok=True)
    with open(os.path.join(UPLOAD_ROOT, "active_params.txt"), "w") as fh:
        fh.write(path)


def camera_or_none():
    try:
        return load_params(active_params_path()), None
    except CalibrationError as exc:
        return None, str(exc)


# --------------------------------------------------------------------------
# Navigation -- one nav bar across every page, so a screen recording can walk
# the whole assignment without retyping URLs.
# --------------------------------------------------------------------------
NAV = [
    ("index", "Overview", ""),
    ("calibration", "Step 1 - Calibration", "Part A"),
    ("measurement", "Step 2 - Measurement", "Part B"),
    ("validation", "Step 3 - Validation", "Part C"),
    ("theory", "Theory - Two Cameras", "Part D"),
]


@app.context_processor
def inject_nav():
    return {"nav": NAV, "active_endpoint": request.endpoint}


# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
@app.route("/")
def index():
    cam, err = camera_or_none()
    csv_rows = 0
    if os.path.exists(DEFAULT_CSV):
        with open(DEFAULT_CSV, newline="") as fh:
            csv_rows = sum(1 for _ in csv.DictReader(fh))
    return render_template("index.html", cam=cam, cam_error=err,
                           params_path=os.path.relpath(active_params_path(), ROOT),
                           csv_rows=csv_rows)


# --------------------------------------------------------------------------
# Step 1 -- calibration
# --------------------------------------------------------------------------
@app.route("/calibration", methods=["GET", "POST"])
def calibration():
    cam, err = camera_or_none()
    context = {
        "cam": cam, "cam_error": err,
        "params_path": os.path.relpath(active_params_path(), ROOT),
        "existing_debug": sorted(os.listdir(CALIB_DEBUG_DIR))[:12]
        if os.path.isdir(CALIB_DEBUG_DIR) else [],
        "pattern_default": "%dx%d" % calib.PATTERN_SIZE,
        "square_default": calib.SQUARE_SIZE_MM,
    }
    if request.method == "GET":
        return render_template("calibration.html", **context)

    files = [f for f in request.files.getlist("images") if f and f.filename]
    if not files:
        context["error"] = "No files selected."
        return render_template("calibration.html", **context), 400

    try:
        pattern_size = calib.parse_pattern(request.form.get("pattern", "9x6"))
        square = float(request.form.get("square_size", calib.SQUARE_SIZE_MM))
    except (ValueError, IndexError):
        context["error"] = "Pattern must look like 9x6 and square size must be a number."
        return render_template("calibration.html", **context), 400
    if square <= 0:
        context["error"] = "Square size must be positive."
        return render_template("calibration.html", **context), 400

    token, sess = session_dir()
    img_dir = os.path.join(sess, "images")
    debug_dir = os.path.join(sess, "debug")
    os.makedirs(img_dir, exist_ok=True)

    saved, skipped = [], []
    for f in files:
        name = secure_filename(f.filename)
        if not allowed_image(name):
            skipped.append(name)
            continue
        path = os.path.join(img_dir, name)
        f.save(path)
        saved.append(path)

    if not saved:
        shutil.rmtree(sess, ignore_errors=True)
        context["error"] = ("None of the uploaded files were images (%s)."
                            % ", ".join(sorted(ALLOWED_IMAGE_EXT)))
        return render_template("calibration.html", **context), 400

    # Same detection + fit the CLI runs -- imported, not reimplemented.
    objpoints, imgpoints, accepted, image_size, rejected = calib.detect_corners(
        sorted(saved), pattern_size, square, debug_dir)

    if len(objpoints) < 3:
        context["error"] = ("Only %d image(s) gave a %dx%d pattern; need at least 3 "
                            "to fit. Check the pattern size and the photos."
                            % (len(objpoints), pattern_size[0], pattern_size[1]))
        context["rejected"] = [(os.path.basename(p), why) for p, why in rejected]
        return render_template("calibration.html", **context), 400

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None)
    errors = calib.per_image_errors(objpoints, imgpoints, rvecs, tvecs, K, dist)

    yaml_path = os.path.join(sess, "camera_params.yaml")
    calib.save_params(yaml_path, K, dist, image_size, pattern_size, square, rms,
                      len(accepted), notes="fitted via the web app")
    set_active_params(yaml_path)

    order = list(np.argsort(errors))
    fitted = load_params(yaml_path)
    context.update({
        "cam": fitted,
        "cam_error": None,
        "params_path": os.path.relpath(yaml_path, ROOT),
        "result": {
            "token": token,
            "rms": float(rms),
            "accepted": len(accepted),
            "uploaded": len(saved),
            "image_size": image_size,
            "skipped": skipped,
            "rejected": [(os.path.basename(p), why) for p, why in rejected],
            "errors": [(os.path.basename(accepted[i]), float(errors[i]))
                       for i in order],
            "mean_error": float(np.mean(errors)),
            "debug_images": sorted(os.listdir(debug_dir))[:12]
            if os.path.isdir(debug_dir) else [],
            "checks": [
                ("15+ images accepted", len(accepted) >= 15),
                ("RMS below 0.5 px", rms < 0.5),
                ("c_x / width near 0.5", 0.45 <= fitted.cx / image_size[0] <= 0.55),
                ("c_y / height near 0.5", 0.45 <= fitted.cy / image_size[1] <= 0.55),
                ("f_y / f_x within 1%", abs(fitted.fy / fitted.fx - 1) < 0.01),
            ],
        },
    })
    return render_template("calibration.html", **context)


@app.route("/calibration/download/<token>")
def download_params(token):
    _, sess = session_dir(token, create=False)
    path = os.path.join(sess, "camera_params.yaml")
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True, download_name="camera_params.yaml")


@app.route("/calibration/debug/<token>/<path:name>")
def calibration_debug_image(token, name):
    _, sess = session_dir(token, create=False)
    return send_from_directory(os.path.join(sess, "debug"), name)


@app.route("/calibration/existing-debug/<path:name>")
def existing_debug_image(name):
    return send_from_directory(CALIB_DEBUG_DIR, name)


@app.route("/calibration/use-default", methods=["POST"])
def use_default_params():
    """Fall back to the calibration produced by the CLI."""
    set_active_params(DEFAULT_PARAMS)
    return redirect(url_for("calibration"))


# --------------------------------------------------------------------------
# Step 2 -- measurement
# --------------------------------------------------------------------------
@app.route("/measurement", methods=["GET", "POST"])
def measurement():
    cam, err = camera_or_none()
    samples = sorted(os.listdir(SAMPLE_MEAS_DIR))[:8] if os.path.isdir(
        SAMPLE_MEAS_DIR) else []
    context = {
        "cam": cam, "cam_error": err,
        "params_path": os.path.relpath(active_params_path(), ROOT),
        "samples": samples,
        "image_url": None, "token": None,
        "nat_w": None, "nat_h": None,
    }
    if request.method == "GET":
        return render_template("measurement.html", **context)

    if cam is None:
        context["error"] = "Calibrate first: no camera parameters are loaded."
        return render_template("measurement.html", **context), 400

    sample = request.form.get("sample")
    token, sess = session_dir()
    if sample:
        name = secure_filename(sample)
        src = os.path.join(SAMPLE_MEAS_DIR, name)
        if not os.path.exists(src):
            abort(404)
        dest = os.path.join(sess, name)
        shutil.copyfile(src, dest)
    else:
        f = request.files.get("image")
        if not f or not f.filename:
            context["error"] = "No image selected."
            return render_template("measurement.html", **context), 400
        name = secure_filename(f.filename)
        if not allowed_image(name):
            context["error"] = "Not an image file."
            return render_template("measurement.html", **context), 400
        dest = os.path.join(sess, name)
        f.save(dest)

    img = cv2.imread(dest, cv2.IMREAD_COLOR)
    if img is None:
        context["error"] = "Could not decode that image."
        return render_template("measurement.html", **context), 400
    nat_h, nat_w = img.shape[:2]

    warning = None
    try:
        _, warning = check_resolution(cam, (nat_w, nat_h),
                                      allow_scaling=bool(request.form.get("allow_scaling")))
        res_error = None
    except CalibrationError as exc:
        res_error = str(exc)

    context.update({
        "image_url": url_for("uploaded_image", token=token, name=os.path.basename(dest)),
        "image_name": os.path.basename(dest),
        "token": token,
        "nat_w": nat_w, "nat_h": nat_h,
        "resolution_warning": warning,
        "resolution_error": res_error,
        "z_default": request.form.get("Z", "2400"),
        "allow_scaling": bool(request.form.get("allow_scaling")),
    })
    return render_template("measurement.html", **context)


@app.route("/uploads/<token>/<path:name>")
def uploaded_image(token, name):
    _, sess = session_dir(token, create=False)
    return send_from_directory(sess, name)


@app.route("/api/measure", methods=["POST"])
def api_measure():
    """Two display-space clicks + Z -> real-world dimensions.

    The browser sends display AND natural dimensions; the conversion to
    full-resolution image coordinates happens here, server-side.
    """
    data = request.get_json(silent=True) or {}
    try:
        token = str(data["token"])
        name = secure_filename(str(data["image_name"]))
        Z = float(data["Z"])
        disp_w, disp_h = float(data["disp_w"]), float(data["disp_h"])
        nat_w, nat_h = float(data["nat_w"]), float(data["nat_h"])
        p1_disp = (float(data["x1"]), float(data["y1"]))
        p2_disp = (float(data["x2"]), float(data["y2"]))
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "malformed request"}), 400

    cam, err = camera_or_none()
    if cam is None:
        return jsonify({"error": err}), 400

    _, sess = session_dir(token, create=False)
    path = os.path.join(sess, name)
    if not os.path.exists(path):
        return jsonify({"error": "image not found; re-upload it"}), 404

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "could not decode the image"}), 400
    real_h, real_w = img.shape[:2]

    try:
        cam_i, warning = check_resolution(
            cam, (real_w, real_h), allow_scaling=bool(data.get("allow_scaling")))
    except CalibrationError as exc:
        return jsonify({"error": str(exc)}), 400

    # Display -> full resolution. Trust the server's own read of the file for
    # the natural size, not the client's claim about it.
    try:
        p1 = scale_click(p1_disp[0], p1_disp[1], disp_w, disp_h, real_w, real_h)
        p2 = scale_click(p2_disp[0], p2_disp[1], disp_w, disp_h, real_w, real_h)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        result = measure(p1, p2, Z, cam_i)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    unc = measure_uncertainty(result, cam_i)
    annot_name = os.path.splitext(name)[0] + "_annotated.jpg"
    cv2.imwrite(os.path.join(sess, annot_name),
                annotate(img, p1, p2, result), [cv2.IMWRITE_JPEG_QUALITY, 90])

    return jsonify({
        "result": result,
        "uncertainty": unc,
        "warning": warning,
        "client_natural": [nat_w, nat_h],
        "server_natural": [real_w, real_h],
        "annotated_url": url_for("uploaded_image", token=token, name=annot_name),
    })


# --------------------------------------------------------------------------
# Step 3 -- validation
# --------------------------------------------------------------------------
@app.route("/validation", methods=["GET", "POST"])
def validation():
    context = {"csv_path": os.path.relpath(DEFAULT_CSV, ROOT), "plots": []}

    rows = None
    if request.method == "POST":
        f = request.files.get("csv")
        if not f or not f.filename:
            context["error"] = "No CSV selected."
            return render_template("validation.html", **context), 400
        try:
            text = f.read().decode("utf-8-sig")
            rows = list(csv.DictReader(io.StringIO(text)))
        except (UnicodeDecodeError, csv.Error) as exc:
            context["error"] = "Could not parse that CSV: %s" % exc
            return render_template("validation.html", **context), 400
        context["csv_path"] = secure_filename(f.filename)
    elif os.path.exists(DEFAULT_CSV):
        with open(DEFAULT_CSV, newline="") as fh:
            rows = list(csv.DictReader(fh))

    if not rows:
        context["error"] = ("No measurements found. Log 20 rows in %s, or generate "
                            "the synthetic stand-in with tools/make_synthetic_data.py "
                            "and tools/simulate_measurements.py."
                            % os.path.relpath(DEFAULT_CSV, ROOT))
        return render_template("validation.html", **context)

    missing = [c for c in analyze.REQUIRED if c not in rows[0]]
    if missing:
        context["error"] = "CSV is missing column(s): %s" % ", ".join(missing)
        return render_template("validation.html", **context), 400

    try:
        gt, meas, Z, err, pct, stats = analyze.compute(rows)
    except (ValueError, KeyError) as exc:
        context["error"] = "Could not compute statistics: %s" % exc
        return render_template("validation.html", **context), 400

    table = []
    for i, r in enumerate(rows):
        table.append({
            "id": r["id"], "object": r["object"], "dimension": r["dimension"],
            "Z": Z[i], "gt": gt[i], "meas": meas[i],
            "err": err[i], "pct": pct[i],
        })

    by_dim = []
    for dim in sorted(set(r["dimension"] for r in rows)):
        m = np.array([r["dimension"] == dim for r in rows])
        e, p = err[m], pct[m]
        by_dim.append({
            "dim": dim, "n": int(m.sum()), "mean": float(e.mean()),
            "std": float(e.std(ddof=1)) if m.sum() > 1 else 0.0,
            "mape": float(np.abs(p).mean()),
        })

    plots = [p for p in ("error_vs_size.png", "error_vs_Z.png",
                         "error_distribution.png")
             if os.path.exists(os.path.join(VALIDATION_OUTPUT, p))]

    context.update({
        "rows": table, "stats": stats, "by_dim": by_dim, "plots": plots,
        "worst": rows[stats["max_abs_error_idx"]],
        "report_text": analyze.report(rows, gt, meas, Z, err, pct, stats),
    })
    return render_template("validation.html", **context)


@app.route("/validation/plot/<path:name>")
def validation_plot(name):
    return send_from_directory(VALIDATION_OUTPUT, name)


# --------------------------------------------------------------------------
# Part D -- theory
# --------------------------------------------------------------------------
@app.route("/theory")
def theory():
    return render_template("theory.html")


@app.errorhandler(413)
def too_large(_exc):
    return render_template("error.html",
                           message="Upload exceeds the %d MB limit."
                                   % MAX_CONTENT_MB), 413


def main(argv=None):
    ap = argparse.ArgumentParser(description="Module 2 web application")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(UPLOAD_ROOT, exist_ok=True)
    print("=" * 66)
    print("CSc 8830 Module 2 -- web application")
    print("=" * 66)
    params = active_params_path()
    if os.path.exists(params):
        cam = load_params(params)
        print("camera   : %s" % os.path.relpath(params, ROOT))
        print("           fx=%.2f fy=%.2f cx=%.2f cy=%.2f @ %dx%d"
              % (cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height))
    else:
        print("camera   : NONE -- run Step 1 first, or fit on the Calibration page")
    print("open     : http://%s:%d" % (args.host, args.port))
    print("=" * 66)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
