"""Module 2 -- camera calibration and real-world measurement, as a blueprint.

THREE PAGES, ONE PER THING YOU ACTUALLY DO
    /module-2/          Calibration  board photos in, K and the distortion out
    /module-2/measure   Measure      photo + Z, click two points, record the result
    /module-2/records   Records      the logged measurements, statistics, figures

    It used to be five: an overview that only restated the navigation bar, one
    page per lettered part, and the Part D derivation. The derivation is a
    written deliverable, not something done in a browser, so it lives in
    theory/two_camera_derivation.md and goes into the PDF from there.

AN UPLOAD IS THE DATASET, NOT AN ADDITION TO IT
    Uploading board photos on Calibration replaces calibration/data/images and
    the corner overlays wholesale, and only after the fit succeeds
    (app/dataset.py stage/install) -- so a batch the corner detector rejects
    leaves the previous data intact instead of emptying the page. Measurement
    photos accumulate in validation/data/images instead, because a measurement
    run is built up one object at a time.

ONE CAMERA FILE, NOT AN ACTIVE-PARAMS POINTER
    calibration/output/camera_params.yaml is the calibration, for this blueprint
    and for the CLI both. A web fit overwrites it. There is deliberately no
    "which of several fits is active" marker: it was a second source of truth
    that could disagree with what the Records page had already been computed
    from.

THE ONE REAL GOTCHA (plan section 25)
    Canvas clicks arrive in DISPLAY coordinates. A 2016-px-wide photo shown in
    an 800-px canvas means every click must be multiplied by 2016/800 = 2.52
    before it reaches the geometry, or every measurement is wrong by that
    factor. The browser posts the canvas size alongside each click and
    scale_click() does the conversion server-side, against the server's own
    read of the file, because K is in pixels at the calibration resolution.

RECORDING IS EXPLICIT, AND THE SERVER RECOMPUTES
    A two-click measurement is not a record until the object, the dimension and
    the ruler ground truth are entered and Record is pressed, so mis-clicks
    never reach the CSV. The form posts the pixel coordinates back and the
    server recomputes measured_mm from them -- the browser's number is never
    trusted -- which is what keeps the CSV internally consistent and
    recomputable by `analyze.py --recompute` after any change to the geometry.

REPORT FIGURES
    Each recorded row also writes a downscaled annotated JPEG to
    validation/output/figures/, sized for a document rather than for a phone
    sensor, and the Records page zips them. The full-resolution original stays
    in validation/data/images/ so a row can be recomputed.

NO GEOMETRY LIVES IN THIS FILE
    Every number comes from week_2/module2/measurement/geometry.py,
    .../calibration/calibrate.py and .../validation/analyze.py -- the same code
    the CLI runs -- so the web app and the CLI cannot disagree. A blueprint that
    recomputes something itself is a bug here even when it agrees.
"""

import csv
import io
import math
import os
import re
import sys
import zipfile

import cv2
import numpy as np
from flask import (Blueprint, abort, jsonify, redirect, render_template,
                   request, send_file, send_from_directory, url_for)
from werkzeug.utils import secure_filename

HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(HERE)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import dataset  # noqa: E402
from assignments import REPO_ROOT, UPLOAD_ROOT, get  # noqa: E402

ASSIGNMENT = get("module-2")
MODULE = ASSIGNMENT.dir                     # <repo>/week_2/module2

for _part in ("measurement", "calibration", "validation"):
    sys.path.insert(0, os.path.join(MODULE, _part))

# geometry.measure is aliased because the Measure page's view function owns the
# name `measure` in this module -- the endpoint is url_for('module2.measure'),
# and renaming the view to dodge a collision would rename the URL with it.
from geometry import (CalibrationError, annotate, check_resolution,  # noqa: E402
                      load_params, measure_uncertainty)
from geometry import measure as measure_span  # noqa: E402
import calibrate as calib  # noqa: E402
import analyze  # noqa: E402


def _m(*parts):
    return os.path.join(MODULE, *parts)


# The assignment's own folders. Uploads land in these, which is what makes a
# replacement persist across restarts.
BOARD_IMAGES = _m("calibration", "data", "images")
BOARD_DEBUG = _m("calibration", "output", "debug")
PARAMS = _m("calibration", "output", "camera_params.yaml")
BOARD_ERROR_CSV = _m("calibration", "output", "reprojection_errors.csv")

MEAS_IMAGES = _m("validation", "data", "images")
RECORDS_CSV = _m("validation", "data", "measurements.csv")
VAL_OUTPUT = _m("validation", "output")
FIGURES = _m("validation", "output", "figures")
STATS_TXT = _m("validation", "output", "statistics.txt")
AUGMENTED_CSV = _m("validation", "output", "measurements_with_errors.csv")

PLOT_NAMES = ("error_vs_size.png", "error_vs_Z.png", "error_distribution.png")

# Session scratch: staging for a calibration fit, and the annotated preview
# shown before a measurement is recorded. Git-ignored; nothing here is source.
SCRATCH = os.path.join(UPLOAD_ROOT, ASSIGNMENT.slug)
PREVIEW = os.path.join(SCRATCH, "preview")

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

CSV_COLUMNS = ["id", "image_file", "object", "dimension", "Z_mm",
               "ground_truth_mm", "measured_mm", "u1", "v1", "u2", "v2"]

# The CSV's dimension codes, and which measured component each one means.
DIMENSIONS = [("W", "W — width, |ΔX|"),
              ("H", "H — height, |ΔY|"),
              ("D", "D — diagonal, the full span")]
MEASURED_KEY = {"W": "width_mm", "H": "height_mm", "D": "length_mm"}

# Wide enough to fill a page in a document, small enough to embed twenty of
# them; a phone original is 3-4x this and bloats the PDF for no visible gain.
FIGURE_MAX_PX = 1400

bp = Blueprint("module2", __name__)

NAV = [
    ("module2.index", "Calibration", ""),
    ("module2.measure", "Measure", ""),
    ("module2.records", "Records", ""),
]


# --------------------------------------------------------------------------
# The section-25 fix. Kept module-level and tiny so it can be imported and
# exercised on its own rather than only through a request.
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


def camera_or_none():
    try:
        return load_params(PARAMS), None
    except CalibrationError as exc:
        return None, str(exc)


def listdir_images(path):
    if not os.path.isdir(path):
        return []
    return sorted(n for n in os.listdir(path) if allowed_image(n))


def safe_name(name):
    """A basename that cannot escape the folder it is joined to."""
    clean = secure_filename(os.path.basename(str(name)))
    if not clean:
        abort(400, "bad filename")
    return clean


# --------------------------------------------------------------------------
# The records CSV -- the one file the Measure and Records pages share
# --------------------------------------------------------------------------
def read_records():
    if not os.path.exists(RECORDS_CSV):
        return []
    with open(RECORDS_CSV, newline="") as fh:
        return [r for r in csv.DictReader(fh) if (r.get("id") or "").strip()]


def write_records(rows):
    os.makedirs(os.path.dirname(RECORDS_CSV), exist_ok=True)
    with open(RECORDS_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def next_record_id(rows):
    used = []
    for r in rows:
        try:
            used.append(int(r["id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return max(used) + 1 if used else 1


def figure_name(row):
    slug = re.sub(r"[^a-z0-9]+", "-", str(row["object"]).lower()).strip("-")
    return "%03d_%s_%s.jpg" % (int(row["id"]), slug or "object",
                               row["dimension"])


def write_report_figure(row, image_path, p1, p2, result):
    """A downscaled annotated JPEG of one recorded measurement.

    The original stays full resolution in validation/data/images so the row can
    be recomputed; this is the copy that goes in the report.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    label = "%s = %.1f mm  (truth %.1f mm)" % (
        row["dimension"], float(row["measured_mm"]),
        float(row["ground_truth_mm"]))
    out = annotate(img, p1, p2, result, label=label)

    h, w = out.shape[:2]
    if max(w, h) > FIGURE_MAX_PX:
        # INTER_AREA is the correct filter for shrinking; the default bilinear
        # aliases the annotation lines into a dashed mess.
        s = FIGURE_MAX_PX / float(max(w, h))
        out = cv2.resize(out, (int(round(w * s)), int(round(h * s))),
                         interpolation=cv2.INTER_AREA)

    os.makedirs(FIGURES, exist_ok=True)
    name = figure_name(row)
    cv2.imwrite(os.path.join(FIGURES, name), out,
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    return name


def refresh_outputs(rows):
    """Recompute statistics.txt, the augmented CSV and the plots from rows.

    Run on every change to the records so the Records page can never show a
    plot drawn from data that has since been edited. Below three rows there is
    nothing to plot, and the stale files are removed rather than left behind.
    """
    if len(rows) < 2:
        dataset.remove(STATS_TXT, AUGMENTED_CSV,
                       *[os.path.join(VAL_OUTPUT, n) for n in PLOT_NAMES])
        return
    try:
        gt, meas, Z, err, pct, stats = analyze.compute(rows)
    except (ValueError, KeyError, ZeroDivisionError):
        return
    os.makedirs(VAL_OUTPUT, exist_ok=True)
    with open(STATS_TXT, "w") as fh:
        fh.write(analyze.report(rows, gt, meas, Z, err, pct, stats))
    analyze.write_augmented_csv(rows, err, pct, AUGMENTED_CSV)
    if len(rows) >= 3:
        try:
            analyze.make_plots(gt, Z, err, pct, rows, VAL_OUTPUT)
        except Exception:       # matplotlib missing, or a backend that is not
            pass                # available headless -- the tables still stand


# --------------------------------------------------------------------------
# Calibration -- the landing page
# --------------------------------------------------------------------------
@bp.route("/", methods=["GET", "POST"])
def index():
    cam, err = camera_or_none()
    context = {
        "cam": cam, "cam_error": err,
        "params_path": os.path.relpath(PARAMS, REPO_ROOT),
        "board_images": listdir_images(BOARD_IMAGES),
        "overlays": listdir_images(BOARD_DEBUG)[:12],
        "pattern_default": "%dx%d" % calib.PATTERN_SIZE,
        "square_default": calib.SQUARE_SIZE_MM,
        "params_exist": os.path.exists(PARAMS),
    }
    if request.method == "GET":
        return render_template("module2/calibration.html", **context)

    files = [f for f in request.files.getlist("images") if f and f.filename]
    if not files:
        context["error"] = "No files selected."
        return render_template("module2/calibration.html", **context), 400

    try:
        pattern_size = calib.parse_pattern(request.form.get("pattern", "9x6"))
        square = float(request.form.get("square_size", calib.SQUARE_SIZE_MM))
    except (ValueError, IndexError):
        context["error"] = ("Pattern must look like 9x6 and the square size "
                            "must be a number.")
        return render_template("module2/calibration.html", **context), 400
    if square <= 0:
        context["error"] = "Square size must be positive."
        return render_template("module2/calibration.html", **context), 400

    # Staged, not installed: the existing dataset survives a failed fit.
    os.makedirs(SCRATCH, exist_ok=True)
    staging = dataset.stage(SCRATCH)
    staged_images = os.path.join(staging, "images")
    staged_debug = os.path.join(staging, "debug")
    os.makedirs(staged_images, exist_ok=True)

    saved, skipped = [], []
    for f in files:
        name = safe_name(f.filename)
        if not allowed_image(name):
            skipped.append(name)
            continue
        path = os.path.join(staged_images, name)
        f.save(path)
        saved.append(path)

    if not saved:
        dataset.discard(staging)
        context["error"] = ("None of the uploaded files were images (%s)."
                            % ", ".join(sorted(ALLOWED_IMAGE_EXT)))
        return render_template("module2/calibration.html", **context), 400

    # Same detection + fit the CLI runs -- imported, not reimplemented.
    objpoints, imgpoints, accepted, image_size, rejected = calib.detect_corners(
        sorted(saved), pattern_size, square, staged_debug)

    if len(objpoints) < 3:
        dataset.discard(staging)
        context["error"] = (
            "Only %d image(s) gave a %dx%d pattern; at least 3 are needed to "
            "fit, so nothing was changed and the previous calibration is still "
            "loaded. Check the pattern size against the printed board."
            % (len(objpoints), pattern_size[0], pattern_size[1]))
        context["rejected"] = [(os.path.basename(p), why) for p, why in rejected]
        return render_template("module2/calibration.html", **context), 400

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None)
    errors = calib.per_image_errors(objpoints, imgpoints, rvecs, tvecs, K, dist)

    # The fit is good: the uploaded boards become the dataset.
    dataset.install(staged_images, BOARD_IMAGES)
    if os.path.isdir(staged_debug):
        dataset.install(staged_debug, BOARD_DEBUG)
    dataset.discard(staging)

    calib.save_params(PARAMS, K, dist, image_size, pattern_size, square, rms,
                      len(accepted), notes="fitted via the web app")
    with open(BOARD_ERROR_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "rms_px"])
        for path, e in zip(accepted, errors):
            w.writerow([os.path.basename(path), "%.6f" % e])

    fitted = load_params(PARAMS)
    order = list(np.argsort(errors))
    context.update({
        "cam": fitted,
        "cam_error": None,
        "params_exist": True,
        "board_images": listdir_images(BOARD_IMAGES),
        "overlays": listdir_images(BOARD_DEBUG)[:12],
        "result": {
            "rms": float(rms),
            "accepted": len(accepted),
            "uploaded": len(saved),
            "image_size": image_size,
            "skipped": skipped,
            "rejected": [(os.path.basename(p), why) for p, why in rejected],
            "errors": [(os.path.basename(accepted[i]), float(errors[i]))
                       for i in order],
            "mean_error": float(np.mean(errors)),
            "checks": [
                ("15+ images accepted", len(accepted) >= 15),
                ("RMS below 0.5 px", rms < 0.5),
                ("c_x / width near 0.5", 0.45 <= fitted.cx / image_size[0] <= 0.55),
                ("c_y / height near 0.5", 0.45 <= fitted.cy / image_size[1] <= 0.55),
                ("f_y / f_x within 1%", abs(fitted.fy / fitted.fx - 1) < 0.01),
            ],
        },
    })
    return render_template("module2/calibration.html", **context)


@bp.route("/params.yaml")
def download_params():
    if not os.path.exists(PARAMS):
        abort(404)
    return send_file(PARAMS, as_attachment=True,
                     download_name="camera_params.yaml")


@bp.route("/overlay/<path:name>")
def overlay_image(name):
    return send_from_directory(BOARD_DEBUG, safe_name(name))


# --------------------------------------------------------------------------
# Measure
# --------------------------------------------------------------------------
def measure_context():
    cam, err = camera_or_none()
    images = listdir_images(MEAS_IMAGES)
    return {
        "cam": cam, "cam_error": err,
        "params_path": os.path.relpath(PARAMS, REPO_ROOT),
        "images": images,
        "dimensions": DIMENSIONS,
        "image_name": None, "nat_w": None, "nat_h": None,
        "z_default": "2400", "allow_scaling": False,
        "record_count": len(read_records()),
    }


@bp.route("/measure", methods=["GET", "POST"])
def measure():
    context = measure_context()

    if request.method == "POST":
        # Post/redirect/get: refreshing the page after an upload must not
        # re-upload the photo, and the loaded image stays linkable.
        scaling = bool(request.form.get("allow_scaling"))
        Z = request.form.get("Z", "2400")

        # A chosen file wins over the dropdown. The dropdown keeps the loaded
        # photo selected, so checking it first would silently ignore the new
        # upload of anyone who measures two photos in a row.
        f = request.files.get("image")
        if f and f.filename:
            name = safe_name(f.filename)
            if not allowed_image(name):
                context["error"] = "Not an image file."
                return render_template("module2/measure.html", **context), 400
            os.makedirs(MEAS_IMAGES, exist_ok=True)
            f.save(os.path.join(MEAS_IMAGES, name))
            return redirect(url_for(".measure", image=name, Z=Z,
                                    scale=1 if scaling else None))

        name = request.form.get("existing")
        if not name:
            context["error"] = "No image selected."
            return render_template("module2/measure.html", **context), 400
        return redirect(url_for(".measure", image=safe_name(name), Z=Z,
                                scale=1 if scaling else None))

    context["z_default"] = request.args.get("Z") or "2400"
    context["allow_scaling"] = bool(request.args.get("scale"))
    context["recorded"] = request.args.get("recorded")

    name = request.args.get("image")
    if not name:
        return render_template("module2/measure.html", **context)

    name = safe_name(name)
    path = os.path.join(MEAS_IMAGES, name)
    if not os.path.exists(path):
        context["error"] = "%s is no longer on disk." % name
        return render_template("module2/measure.html", **context), 404

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        context["error"] = "Could not decode %s." % name
        return render_template("module2/measure.html", **context), 400
    nat_h, nat_w = img.shape[:2]

    warning, res_error = None, None
    if context["cam"] is not None:
        try:
            _, warning = check_resolution(
                context["cam"], (nat_w, nat_h),
                allow_scaling=context["allow_scaling"])
        except CalibrationError as exc:
            res_error = str(exc)

    context.update({
        "image_name": name,
        "image_url": url_for(".measurement_image", name=name),
        "nat_w": nat_w, "nat_h": nat_h,
        "resolution_warning": warning,
        "resolution_error": res_error,
    })
    return render_template("module2/measure.html", **context)


@bp.route("/photo/<path:name>")
def measurement_image(name):
    return send_from_directory(MEAS_IMAGES, safe_name(name))


@bp.route("/preview/<path:name>")
def preview_image(name):
    return send_from_directory(PREVIEW, safe_name(name))


@bp.route("/api/measure", methods=["POST"])
def api_measure():
    """Two display-space clicks + Z -> real-world dimensions.

    The browser sends the canvas size; the conversion to full-resolution image
    coordinates happens here, against the server's own read of the file.
    """
    data = request.get_json(silent=True) or {}
    try:
        name = safe_name(data["image_name"])
        Z = float(data["Z"])
        disp_w, disp_h = float(data["disp_w"]), float(data["disp_h"])
        p1_disp = (float(data["x1"]), float(data["y1"]))
        p2_disp = (float(data["x2"]), float(data["y2"]))
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "malformed request"}), 400

    cam, err = camera_or_none()
    if cam is None:
        return jsonify({"error": err}), 400

    path = os.path.join(MEAS_IMAGES, name)
    if not os.path.exists(path):
        return jsonify({"error": "image not found; re-upload it"}), 404

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "could not decode the image"}), 400
    real_h, real_w = img.shape[:2]

    try:
        cam_i, warning = check_resolution(
            cam, (real_w, real_h),
            allow_scaling=bool(data.get("allow_scaling")))
    except CalibrationError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        p1 = scale_click(p1_disp[0], p1_disp[1], disp_w, disp_h, real_w, real_h)
        p2 = scale_click(p2_disp[0], p2_disp[1], disp_w, disp_h, real_w, real_h)
        result = measure_span(p1, p2, Z, cam_i)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    os.makedirs(PREVIEW, exist_ok=True)
    annot = os.path.splitext(name)[0] + "_preview.jpg"
    cv2.imwrite(os.path.join(PREVIEW, annot), annotate(img, p1, p2, result),
                [cv2.IMWRITE_JPEG_QUALITY, 88])

    return jsonify({
        "result": result,
        "uncertainty": measure_uncertainty(result, cam_i),
        "warning": warning,
        "annotated_url": url_for(".preview_image", name=annot),
    })


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------
@bp.route("/records/add", methods=["POST"])
def record_add():
    """Append one measurement to the CSV and write its report figure.

    measured_mm is recomputed here from the posted pixel coordinates rather
    than taken from the browser, so the CSV is always consistent with the
    calibration and with geometry.py.
    """
    form = request.form
    try:
        name = safe_name(form["image_file"])
        Z = float(form["Z_mm"])
        ground_truth = float(form["ground_truth_mm"])
        dimension = form["dimension"]
        p1 = (float(form["u1"]), float(form["v1"]))
        p2 = (float(form["u2"]), float(form["v2"]))
    except (KeyError, TypeError, ValueError):
        abort(400, "the record form was incomplete")

    obj = (form.get("object") or "").strip()
    if not obj:
        abort(400, "name the object being measured")
    if dimension not in MEASURED_KEY:
        abort(400, "dimension must be one of W, H, D")
    if ground_truth <= 0:
        abort(400, "the ruler ground truth must be positive")

    cam, err = camera_or_none()
    if cam is None:
        abort(400, err)

    path = os.path.join(MEAS_IMAGES, name)
    if not os.path.exists(path):
        abort(404)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        abort(400, "could not decode %s" % name)
    real_h, real_w = img.shape[:2]
    try:
        cam_i, _ = check_resolution(
            cam, (real_w, real_h),
            allow_scaling=bool(form.get("allow_scaling")))
        result = measure_span(p1, p2, Z, cam_i)
    except (CalibrationError, ValueError) as exc:
        abort(400, str(exc))

    rows = read_records()
    row = {
        "id": next_record_id(rows),
        "image_file": name,
        "object": obj,
        "dimension": dimension,
        "Z_mm": "%.1f" % Z,
        "ground_truth_mm": "%.2f" % ground_truth,
        "measured_mm": "%.2f" % result[MEASURED_KEY[dimension]],
        "u1": "%.2f" % p1[0], "v1": "%.2f" % p1[1],
        "u2": "%.2f" % p2[0], "v2": "%.2f" % p2[1],
    }
    write_report_figure(row, path, p1, p2, result)
    rows.append(row)
    write_records(rows)
    refresh_outputs(rows)

    return redirect(url_for(".measure", image=name, Z=form["Z_mm"],
                            scale=1 if form.get("allow_scaling") else None,
                            recorded=row["id"]))


@bp.route("/records/delete", methods=["POST"])
def record_delete():
    rows = read_records()
    target = str(request.form.get("id", "")).strip()
    keep = [r for r in rows if str(r.get("id")) != target]
    if len(keep) == len(rows):
        abort(404)
    gone = [r for r in rows if str(r.get("id")) == target]
    for r in gone:
        try:
            dataset.remove(os.path.join(FIGURES, figure_name(r)))
        except (KeyError, ValueError):
            pass
    write_records(keep)
    refresh_outputs(keep)
    return redirect(url_for(".records", deleted=target))


@bp.route("/records/csv")
def records_csv():
    if not os.path.exists(RECORDS_CSV):
        abort(404)
    return send_file(RECORDS_CSV, as_attachment=True,
                     download_name="measurements.csv")


@bp.route("/records/figures.zip")
def figures_zip():
    names = listdir_images(FIGURES)
    if not names:
        abort(404)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            zf.write(os.path.join(FIGURES, name), arcname=name)
        if os.path.exists(STATS_TXT):
            zf.write(STATS_TXT, arcname="statistics.txt")
        if os.path.exists(RECORDS_CSV):
            zf.write(RECORDS_CSV, arcname="measurements.csv")
        for plot in PLOT_NAMES:
            path = os.path.join(VAL_OUTPUT, plot)
            if os.path.exists(path):
                zf.write(path, arcname=plot)
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name="module2_report_figures.zip")


@bp.route("/records/figure/<path:name>")
def report_figure(name):
    return send_from_directory(FIGURES, safe_name(name))


@bp.route("/records/plot/<path:name>")
def validation_plot(name):
    return send_from_directory(VAL_OUTPUT, safe_name(name))


@bp.route("/records", methods=["GET", "POST"])
def records():
    context = {
        "csv_path": os.path.relpath(RECORDS_CSV, REPO_ROOT),
        "plots": [], "figures": [], "deleted": request.args.get("deleted"),
    }

    if request.method == "POST":
        f = request.files.get("csv")
        if not f or not f.filename:
            context["error"] = "No CSV selected."
            return render_template("module2/records.html", **context), 400
        try:
            text = f.read().decode("utf-8-sig")
            uploaded = list(csv.DictReader(io.StringIO(text)))
        except (UnicodeDecodeError, csv.Error) as exc:
            context["error"] = "Could not parse that CSV: %s" % exc
            return render_template("module2/records.html", **context), 400
        missing = [c for c in analyze.REQUIRED
                   if not uploaded or c not in uploaded[0]]
        if missing:
            context["error"] = ("CSV is missing column(s): %s"
                                % ", ".join(missing))
            return render_template("module2/records.html", **context), 400
        write_records(uploaded)
        refresh_outputs(uploaded)
        return redirect(url_for(".records"))

    rows = read_records()

    if not rows:
        context["empty"] = True
        return render_template("module2/records.html", **context)

    missing = [c for c in analyze.REQUIRED if c not in rows[0]]
    if missing:
        context["error"] = "CSV is missing column(s): %s" % ", ".join(missing)
        return render_template("module2/records.html", **context), 400

    try:
        gt, meas, Z, err, pct, stats = analyze.compute(rows)
    except (ValueError, KeyError) as exc:
        context["error"] = "Could not compute statistics: %s" % exc
        return render_template("module2/records.html", **context), 400

    figures = {}
    for r in rows:
        try:
            name = figure_name(r)
        except (KeyError, ValueError):
            continue
        if os.path.exists(os.path.join(FIGURES, name)):
            figures[str(r["id"])] = name

    table = []
    for i, r in enumerate(rows):
        table.append({
            "id": r["id"], "object": r["object"], "dimension": r["dimension"],
            "image_file": r.get("image_file", ""),
            "Z": Z[i], "gt": gt[i], "meas": meas[i],
            "err": err[i], "pct": pct[i],
            "figure": figures.get(str(r["id"])),
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

    # corrcoef is nan when a column has no spread -- every shot at one Z, or
    # one repeated object size. That is a normal way to run the experiment, so
    # the row is dropped rather than printed as "nan".
    trends = [
        (label, stats[key], note) for key, label, note in (
            ("corr_err_vs_gt", "corr(ground-truth size, signed error)",
             "Strong ⇒ error grows with size: a scale error in f_x or Z, "
             "not click precision."),
            ("corr_pct_vs_Z", "corr(Z, percent error)",
             "A trend implicates Z: a fixed origin offset dZ gives a percent "
             "error dZ/Z that shrinks with distance."))
        if isinstance(stats.get(key), float) and math.isfinite(stats[key])
    ]

    context.update({
        "rows": table, "stats": stats, "by_dim": by_dim, "trends": trends,
        "worst": rows[stats["max_abs_error_idx"]],
        "figures": [f for f in (r["figure"] for r in table) if f],
        "plots": [p for p in PLOT_NAMES
                  if os.path.exists(os.path.join(VAL_OUTPUT, p))],
        "report_text": analyze.report(rows, gt, meas, Z, err, pct, stats),
        "stale_calibration": (os.path.exists(PARAMS) and
                              os.path.exists(RECORDS_CSV) and
                              os.path.getmtime(PARAMS) >
                              os.path.getmtime(RECORDS_CSV)),
    })
    return render_template("module2/records.html", **context)
