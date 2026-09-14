"""
Step 2 -- click-to-measure CLI. Thin UI wrapper; all geometry lives in geometry.py.

HOW TO RUN
    # from module2/
    # interactive: click two points, press 'm' is not needed -- it measures on
    # the second click. r = reset, s = save, q/ESC = quit.
    python measurement/measure_cli.py --image path/to/photo.jpg --Z 2400

    # non-interactive (scriptable, and what the validation harness uses):
    python measurement/measure_cli.py --image photo.jpg --Z 2400 \
        --points 120,340 980,352 --json

    # if the photo resolution differs from calibration, it REFUSES unless:
    python measurement/measure_cli.py --image photo.jpg --Z 2400 --allow-scaling

    # Step 3: measure AND append the row to the measurements CSV in one go,
    # so no pixel coordinate is ever retyped by hand.
    python measurement/measure_cli.py --image IMG_1042.jpg --Z 2380 \
        --log --object "hardback book" --dimension H --truth 240.0
    # interactive + logging: press 's' to save the annotated image and log the row.

OPTIONS
    --params  camera_params.yaml from Step 1 (default calibration/output/...)
    --no-undistort   use the raw pinhole inverse instead of undistortPoints
    --out     annotated image path (default measurement/output/<name>_annotated.jpg)
    --log [CSV]      append the result to the Step 3 CSV
                     (default validation/data/measurements.csv); needs
                     --object, --dimension {W,H,D} and --truth <mm>
    --id N           force the row id (default: one past the highest in the CSV)

An annotated image is always saved; those go straight into the PDF.
"""

import argparse
import csv
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import (CalibrationError, annotate, check_resolution,  # noqa: E402
                      load_params, measure, measure_uncertainty)

DEFAULT_PARAMS = "calibration/output/camera_params.yaml"
DEFAULT_OUT_DIR = "measurement/output"
DEFAULT_LOG_CSV = "validation/data/measurements.csv"
WINDOW = "Step 2 - click two points"
MAX_DISPLAY = 1100  # display downscale cap; clicks are scaled back up


def print_result(result, cam, unc=None):
    print("\n" + "=" * 58)
    print("MEASUREMENT")
    print("=" * 58)
    print("p1 (px)          : (%.1f, %.1f)" % result["p1_px"])
    print("p2 (px)          : (%.1f, %.1f)" % result["p2_px"])
    print("pixel span       : %.2f px  (du=%.2f, dv=%.2f)"
          % (result["span_px"], result["du_px"], result["dv_px"]))
    print("Z                : %.1f mm" % result["Z_mm"])
    print("-" * 58)
    print("WIDTH   (|dX|)   : %9.2f mm" % result["width_mm"])
    print("HEIGHT  (|dY|)   : %9.2f mm" % result["height_mm"])
    print("LENGTH  (diag)   : %9.2f mm" % result["length_mm"])
    print("-" * 58)
    print("scale at this Z  : %.4f mm/px" % result["mm_per_px"])
    print("distortion       : %s"
          % ("undistortPoints (Option 1)" if result["undistorted_points"]
             else "none (raw pinhole inverse)"))
    if unc:
        print("-" * 58)
        print("error budget (1-sigma, plan section 12)")
        print("  click precision : %6.2f %%" % (unc["rel_click"] * 100))
        print("  Z measurement   : %6.2f %%" % (unc["rel_Z"] * 100))
        print("  focal length    : %6.2f %%" % (unc["rel_f"] * 100))
        print("  combined        : %6.2f %%  (+/- %.2f mm)"
              % (unc["rel_total"] * 100, unc["abs_total_mm"]))
        print("  dominant term   : %s" % unc["dominant"])
    print("=" * 58)


def run_interactive(img, cam, Z, undistort, out_path, log_cb=None):
    """cv2 click interface. Two clicks -> measurement."""
    h, w = img.shape[:2]
    scale = min(1.0, MAX_DISPLAY / float(max(w, h)))
    disp_base = cv2.resize(img, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA) if scale < 1.0 else img.copy()

    state = {"pts": [], "result": None, "disp": disp_base.copy()}

    def redraw():
        state["disp"] = disp_base.copy()
        for pt in state["pts"]:
            d = (int(pt[0] * scale), int(pt[1] * scale))
            cv2.circle(state["disp"], d, 5, (0, 0, 255), -1, cv2.LINE_AA)
        if len(state["pts"]) == 2:
            a, b = state["pts"]
            cv2.line(state["disp"],
                     (int(a[0] * scale), int(a[1] * scale)),
                     (int(b[0] * scale), int(b[1] * scale)),
                     (0, 215, 255), 2, cv2.LINE_AA)
            if state["result"]:
                cv2.putText(state["disp"], "%.1f mm" % state["result"]["length_mm"],
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(state["disp"], "%.1f mm" % state["result"]["length_mm"],
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 255, 255), 2, cv2.LINE_AA)

    def on_mouse(event, x, y, flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(state["pts"]) >= 2:
            state["pts"] = []
            state["result"] = None
            state["logged"] = False
        # Display coords -> full-resolution image coords. K was fitted at full
        # resolution, so the geometry must never see display coordinates.
        state["pts"].append((x / scale, y / scale))
        if len(state["pts"]) == 2:
            try:
                state["result"] = measure(state["pts"][0], state["pts"][1],
                                          Z, cam, undistort=undistort)
                print_result(state["result"], cam,
                             measure_uncertainty(state["result"], cam))
            except ValueError as exc:
                print("rejected: %s" % exc)
                state["pts"] = []
        redraw()

    print("\nclick two points. keys: r=reset  s=save%s  q/ESC=quit"
          % ("+log" if log_cb else ""))
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    redraw()

    while True:
        cv2.imshow(WINDOW, state["disp"])
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("r"):
            state["pts"], state["result"] = [], None
            state["logged"] = False
            redraw()
        if key == ord("s") and state["result"]:
            save_annotated(img, state["pts"], state["result"], out_path)
            if log_cb and not state.get("logged"):
                log_cb(state["result"])
                state["logged"] = True
    cv2.destroyAllWindows()
    return state["result"]


def save_annotated(img, pts, result, out_path, stream=None):
    """Write the annotated image. With --json the notice goes to stderr so
    stdout stays pure JSON and stays pipeable."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out = annotate(img, pts[0], pts[1], result)
    cv2.imwrite(out_path, out)
    print("saved annotated image -> %s" % out_path, file=stream or sys.stdout)


# Step 3 log schema (plan section 15). The pixel coordinates are stored so any
# row can be recomputed later with analyze.py --recompute -- without reshooting.
LOG_FIELDS = ["id", "image_file", "object", "dimension", "Z_mm",
              "ground_truth_mm", "measured_mm", "u1", "v1", "u2", "v2"]


def next_log_id(path):
    """One past the highest existing id, so appends never collide."""
    if not os.path.exists(path):
        return 1
    try:
        with open(path, newline="") as fh:
            ids = [int(r["id"]) for r in csv.DictReader(fh)
                   if (r.get("id") or "").strip().lstrip("-").isdigit()]
        return max(ids) + 1 if ids else 1
    except (csv.Error, KeyError, ValueError, OSError):
        return 1


def log_row(path, image_file, obj, dimension, truth_mm, result, row_id=None):
    """Append one Step 3 measurement row, writing a header if the file is new."""
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fresh = (not os.path.exists(path)) or os.path.getsize(path) == 0
    row_id = next_log_id(path) if row_id is None else row_id
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if fresh:
            writer.writeheader()
        writer.writerow({
            "id": row_id,
            "image_file": os.path.basename(image_file),
            "object": obj,
            "dimension": dimension,
            "Z_mm": "%.1f" % result["Z_mm"],
            "ground_truth_mm": "%.2f" % truth_mm,
            "measured_mm": "%.2f" % result["length_mm"],
            "u1": "%.2f" % result["p1_px"][0], "v1": "%.2f" % result["p1_px"][1],
            "u2": "%.2f" % result["p2_px"][0], "v2": "%.2f" % result["p2_px"][1],
        })
    err = result["length_mm"] - truth_mm
    print("logged row id=%d to %s" % (row_id, path))
    print("  %s %s: measured %.2f mm vs truth %.2f mm  (%+.2f mm, %+.2f%%)"
          % (obj, dimension, result["length_mm"], truth_mm, err,
             err / truth_mm * 100 if truth_mm else float("nan")))
    return row_id


def parse_point(text):
    u, v = text.split(",")
    return (float(u), float(v))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 2: measure real-world 2D size")
    ap.add_argument("--image", required=True, help="photo to measure")
    ap.add_argument("--Z", type=float, required=True,
                    help="perpendicular distance from the optical centre to the "
                         "object plane, in mm")
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    ap.add_argument("--points", nargs=2, metavar=("U1,V1", "U2,V2"),
                    help="non-interactive: two pixel coordinates")
    ap.add_argument("--out", default=None, help="annotated output image path")
    ap.add_argument("--no-undistort", action="store_true",
                    help="use the raw pinhole inverse instead of undistortPoints")
    ap.add_argument("--allow-scaling", action="store_true",
                    help="rescale K if the image resolution differs (approximate)")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")

    log = ap.add_argument_group(
        "Step 3 logging",
        "Append the result straight to the measurements CSV instead of "
        "retyping it. Requires --object, --dimension and --truth.")
    log.add_argument("--log", nargs="?", const=DEFAULT_LOG_CSV, default=None,
                     metavar="CSV",
                     help="append a row to this CSV (default %s)" % DEFAULT_LOG_CSV)
    log.add_argument("--object", help="what was measured, e.g. \"hardback book\"")
    log.add_argument("--dimension", choices=["W", "H", "D"],
                     help="which dimension: W, H or D")
    log.add_argument("--truth", type=float,
                     help="ruler ground truth in mm, measured BEFORE imaging")
    log.add_argument("--id", type=int, default=None,
                     help="row id (default: one past the highest in the CSV)")
    args = ap.parse_args(argv)

    if args.log:
        missing = [n for n, v in (("--object", args.object),
                                  ("--dimension", args.dimension),
                                  ("--truth", args.truth)) if v in (None, "")]
        if missing:
            ap.error("--log also needs %s" % ", ".join(missing))
        if args.truth <= 0:
            ap.error("--truth must be a positive length in mm")

    try:
        cam = load_params(args.params)
    except CalibrationError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        print("ERROR: could not read image %s" % args.image, file=sys.stderr)
        return 2
    h, w = img.shape[:2]

    try:
        cam, warning = check_resolution(cam, (w, h), allow_scaling=args.allow_scaling)
    except CalibrationError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2
    if warning:
        print("!" * 58)
        print("WARNING: %s" % warning)
        print("!" * 58)

    if not args.json:
        print("camera   : %s" % args.params)
        print("  fx=%.2f fy=%.2f cx=%.2f cy=%.2f  @ %dx%d  (RMS %.4f px)"
              % (cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height,
                 cam.rms or float("nan")))
        print("image    : %s  (%dx%d)" % (args.image, w, h))
        print("Z        : %.1f mm" % args.Z)

    base = os.path.splitext(os.path.basename(args.image))[0]
    out_path = args.out or os.path.join(DEFAULT_OUT_DIR, base + "_annotated.jpg")
    undistort = not args.no_undistort

    if args.points:
        p1, p2 = parse_point(args.points[0]), parse_point(args.points[1])
        try:
            result = measure(p1, p2, args.Z, cam, undistort=undistort)
        except ValueError as exc:
            print("ERROR: %s" % exc, file=sys.stderr)
            return 2
        unc = measure_uncertainty(result, cam)
        save_annotated(img, (p1, p2), result, out_path,
                       stream=sys.stderr if args.json else None)
        if args.json:
            print(json.dumps({"result": result, "uncertainty": unc,
                              "annotated_image": out_path}, indent=2))
        else:
            print_result(result, cam, unc)
        if args.log:
            log_row(args.log, args.image, args.object, args.dimension,
                    args.truth, result, args.id)
        return 0

    log_cb = None
    if args.log:
        def log_cb(res):
            return log_row(args.log, args.image, args.object, args.dimension,
                           args.truth, res, args.id)
        print("logging enabled -> %s  (press 's' to save + log)" % args.log)

    result = run_interactive(img, cam, args.Z, undistort, out_path, log_cb)
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
