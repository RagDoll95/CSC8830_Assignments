"""
Step 1 -- Camera calibration (Zhang's multi-plane method via cv2.calibrateCamera).

WHAT IT DOES
    Detects 9x6 inner checkerboard corners in every image under data/images/,
    refines them to subpixel accuracy, fits intrinsics K + distortion
    coefficients, prints a per-image reprojection-error table, and writes
    output/camera_params.yaml for Steps 2 and 3 to load.

HOW TO RUN
    # from module2/
    python calibration/calibrate.py

    # with options
    python calibration/calibrate.py \
        --images "calibration/data/images/*.jpg" \
        --square-size 25.0 \
        --pattern 9x6 \
        --out calibration/output/camera_params.yaml

OUTPUTS
    calibration/output/camera_params.yaml   K, dist, image size, RMS, metadata
    calibration/output/debug/*.jpg          corner overlays (one per accepted image)
    calibration/output/reprojection_errors.csv

EXIT CRITERIA (see plan section 5)
    >= 15 images accepted, RMS < ~0.5 px, c_x/width and c_y/height near 0.5.
"""

import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Config -- single source of truth. The README quotes these values.
# --------------------------------------------------------------------------
PATTERN_SIZE = (9, 6)          # inner corners (cols, rows)
SQUARE_SIZE_MM = 25.0          # measure your printed board and set this
IMAGE_GLOB = "calibration/data/images/*"
OUTPUT_YAML = "calibration/output/camera_params.yaml"
DEBUG_DIR = "calibration/output/debug"
ERROR_CSV = "calibration/output/reprojection_errors.csv"

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# cornerSubPix: 11x11 search window, 30 iterations, eps 0.001.
# Skipping this refinement is the most common cause of a mediocre RMS.
SUBPIX_WINDOW = (11, 11)
SUBPIX_ZERO_ZONE = (-1, -1)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE


def build_object_points(pattern_size, square_size_mm):
    """(N,3) float32 template of (col*s, row*s, 0).

    The board defines the world frame with Z = 0. Scaling by the real square
    size in mm is what puts K on a metric scale, so Step 2/3 outputs come out
    in mm rather than in board units.
    """
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return objp * float(square_size_mm)


def list_images(pattern):
    files = sorted(p for p in glob.glob(pattern)
                   if os.path.splitext(p)[1].lower() in VALID_EXT)
    return files


def detect_corners(paths, pattern_size, square_size_mm, debug_dir=None):
    """Detect + refine corners in every image.

    Returns (objpoints, imgpoints, accepted_paths, image_size, rejected).
    Failures are logged and skipped, never fatal -- it matters which shots
    were rejected, so the caller reports them.
    """
    objp_template = build_object_points(pattern_size, square_size_mm)
    objpoints, imgpoints, accepted, rejected = [], [], [], []
    image_size = None

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            rejected.append((path, "unreadable"))
            print("  [SKIP] %-40s unreadable" % os.path.basename(path))
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = (gray.shape[1], gray.shape[0])  # (w, h)

        # Every image must share one resolution: K is resolution-specific.
        if image_size is None:
            image_size = shape
        elif shape != image_size:
            rejected.append((path, "resolution %dx%d != %dx%d"
                             % (shape[0], shape[1], image_size[0], image_size[1])))
            print("  [SKIP] %-40s resolution %dx%d, expected %dx%d"
                  % (os.path.basename(path), shape[0], shape[1],
                     image_size[0], image_size[1]))
            continue

        found, corners = cv2.findChessboardCorners(gray, pattern_size, FIND_FLAGS)
        if not found:
            rejected.append((path, "corners not found"))
            print("  [SKIP] %-40s corners not found" % os.path.basename(path))
            continue

        corners = cv2.cornerSubPix(gray, corners, SUBPIX_WINDOW,
                                   SUBPIX_ZERO_ZONE, SUBPIX_CRITERIA)
        objpoints.append(objp_template.copy())
        imgpoints.append(corners)
        accepted.append(path)
        print("  [ OK ] %-40s %d corners" % (os.path.basename(path), len(corners)))

        if debug_dir:
            overlay = img.copy()
            cv2.drawChessboardCorners(overlay, pattern_size, corners, found)
            cv2.imwrite(os.path.join(debug_dir, os.path.basename(path)), overlay)

    return objpoints, imgpoints, accepted, image_size, rejected


def per_image_errors(objpoints, imgpoints, rvecs, tvecs, K, dist):
    """L2 reprojection error per image, via projectPoints.

    Finds the one bad photo dragging the fit down.
    """
    errors = []
    for i, objp in enumerate(objpoints):
        proj, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], K, dist)
        proj = proj.reshape(-1, 2)
        obs = imgpoints[i].reshape(-1, 2)
        # RMS over the corners of this image.
        err = float(np.sqrt(np.mean(np.sum((proj - obs) ** 2, axis=1))))
        errors.append(err)
    return errors


def save_params(path, K, dist, image_size, pattern_size, square_size_mm,
                rms, n_images, notes=""):
    """Write calibration to YAML with cv2.FileStorage.

    Step 2 loads this file. K is never hardcoded in two places.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", K)
    fs.write("distortion_coefficients", dist)
    fs.write("image_width", int(image_size[0]))
    fs.write("image_height", int(image_size[1]))
    fs.write("square_size_mm", float(square_size_mm))
    fs.write("pattern_cols", int(pattern_size[0]))
    fs.write("pattern_rows", int(pattern_size[1]))
    fs.write("rms_reprojection_error_px", float(rms))
    fs.write("image_count", int(n_images))
    fs.write("notes", notes)
    fs.release()


def print_summary(K, dist, image_size, rms, n_images):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    w, h = image_size

    print("\n" + "=" * 62)
    print("CALIBRATION SUMMARY")
    print("=" * 62)
    print("images used              : %d" % n_images)
    print("resolution               : %d x %d" % (w, h))
    print("RMS reprojection error   : %.4f px" % rms)
    print("f_x, f_y                 : %.2f, %.2f px" % (fx, fy))
    print("f_y / f_x                : %.5f   (square pixels => ~1.0)" % (fy / fx))
    print("c_x, c_y                 : %.2f, %.2f px" % (cx, cy))
    print("c_x / width              : %.4f   (expect ~0.5)" % (cx / w))
    print("c_y / height             : %.4f   (expect ~0.5)" % (cy / h))
    print("distortion [k1 k2 p1 p2 k3]:")
    print("  " + np.array2string(dist.ravel(), precision=6, suppress_small=False))
    print("-" * 62)

    # Exit criteria from the plan, checked automatically.
    checks = [
        ("15+ images accepted", n_images >= 15),
        ("RMS < 0.5 px", rms < 0.5),
        ("c_x/width in [0.45, 0.55]", 0.45 <= cx / w <= 0.55),
        ("c_y/height in [0.45, 0.55]", 0.45 <= cy / h <= 0.55),
        ("f_y/f_x within 1%", abs(fy / fx - 1.0) < 0.01),
    ]
    print("EXIT CRITERIA")
    for label, ok in checks:
        print("  [%s] %s" % ("PASS" if ok else "WARN", label))
    print("=" * 62)
    return all(ok for _, ok in checks)


def parse_pattern(text):
    cols, rows = text.lower().split("x")
    return (int(cols), int(rows))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 1: camera calibration")
    ap.add_argument("--images", default=IMAGE_GLOB, help="glob for calibration images")
    ap.add_argument("--pattern", default="%dx%d" % PATTERN_SIZE,
                    help="inner corners as COLSxROWS (default 9x6)")
    ap.add_argument("--square-size", type=float, default=SQUARE_SIZE_MM,
                    help="checkerboard square size in mm")
    ap.add_argument("--out", default=OUTPUT_YAML, help="output YAML path")
    ap.add_argument("--debug-dir", default=DEBUG_DIR, help="corner-overlay output dir")
    ap.add_argument("--error-csv", default=ERROR_CSV, help="per-image error table")
    ap.add_argument("--notes", default="", help="free-text metadata (phone model, app)")
    args = ap.parse_args(argv)

    pattern_size = parse_pattern(args.pattern)
    paths = list_images(args.images)

    print("Step 1 -- Camera calibration (Zhang's method)")
    print("pattern     : %dx%d inner corners" % pattern_size)
    print("square size : %.3f mm" % args.square_size)
    print("images glob : %s  (%d files)\n" % (args.images, len(paths)))

    if not paths:
        print("ERROR: no images matched %r.\n"
              "       Put 15-20 checkerboard photos in calibration/data/images/,\n"
              "       or generate a synthetic set:  python tools/make_synthetic_data.py"
              % args.images, file=sys.stderr)
        return 2

    objpoints, imgpoints, accepted, image_size, rejected = detect_corners(
        paths, pattern_size, args.square_size, args.debug_dir)

    print("\naccepted %d / %d images" % (len(accepted), len(paths)))
    if rejected:
        print("rejected:")
        for path, why in rejected:
            print("  %-40s %s" % (os.path.basename(path), why))

    if len(objpoints) < 3:
        print("\nERROR: need at least 3 accepted images to fit (got %d)."
              % len(objpoints), file=sys.stderr)
        return 2
    if len(objpoints) < 15:
        print("\nWARNING: only %d images accepted; the plan asks for 15+."
              % len(objpoints))

    # Zhang's multi-plane calibration. Preferred over a DLT fit because DLT
    # models no radial distortion, cannot impose constraints such as a known
    # focal length, and does not minimize geometric reprojection error.
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None)

    errors = per_image_errors(objpoints, imgpoints, rvecs, tvecs, K, dist)

    print("\nPER-IMAGE REPROJECTION ERROR (sorted, worst last)")
    print("  %-42s %10s" % ("image", "RMS px"))
    order = np.argsort(errors)
    for i in order:
        print("  %-42s %10.4f" % (os.path.basename(accepted[i]), errors[i]))
    print("  %-42s %10.4f" % ("-- mean --", float(np.mean(errors))))

    os.makedirs(os.path.dirname(os.path.abspath(args.error_csv)), exist_ok=True)
    with open(args.error_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "rms_reprojection_error_px"])
        for i in order:
            w.writerow([os.path.basename(accepted[i]), "%.6f" % errors[i]])

    notes = args.notes or ("calibrated from %d images at %dx%d"
                           % (len(accepted), image_size[0], image_size[1]))
    save_params(args.out, K, dist, image_size, pattern_size,
                args.square_size, rms, len(accepted), notes)

    print_summary(K, dist, image_size, rms, len(accepted))
    print("\nwrote %s" % args.out)
    print("wrote %s" % args.error_csv)
    print("wrote corner overlays to %s/" % args.debug_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
