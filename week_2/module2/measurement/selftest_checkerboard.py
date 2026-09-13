"""
Step 2 -- end-to-end pipeline self-test against the checkerboard (plan section 11).

WHY THIS EXISTS
    The board's true square size in mm is already known, so measuring the board
    itself validates calibration + backprojection + units in one shot, BEFORE
    20 experimental measurements are burned.

        25 mm square returns ~25.3 mm  -> pipeline is working
        25 mm square returns ~31 mm    -> resolution mismatch or a units bug

HOW TO RUN
    # from module2/
    # Photograph the board fronto-parallel at a tape-measured distance, then:
    python measurement/selftest_checkerboard.py --image board_at_1m.jpg --Z 1000

    # options
    --params        camera_params.yaml (default calibration/output/...)
    --square-size   override the true square size in mm (else read from YAML)
    --tolerance     max acceptable mean error in percent (default 5.0)

WHAT IT MEASURES
    It detects the 9x6 corners, then measures every adjacent corner-to-corner
    gap along both board axes and compares each to the true square size. That
    is ~85 independent measurements from one photo, so the mean and spread are
    meaningful rather than anecdotal.
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "calibration"))
from geometry import (CalibrationError, check_resolution, load_params,  # noqa: E402
                      measure)

DEFAULT_PARAMS = "calibration/output/camera_params.yaml"
FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def adjacent_gaps(corners, pattern_size):
    """Yield (p1, p2, axis) for every adjacent corner pair on the board grid."""
    cols, rows = pattern_size
    grid = corners.reshape(rows, cols, 2)
    for r in range(rows):
        for c in range(cols - 1):
            yield grid[r, c], grid[r, c + 1], "horizontal"
    for r in range(rows - 1):
        for c in range(cols):
            yield grid[r, c], grid[r + 1, c], "vertical"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 2 pipeline self-test on the board")
    ap.add_argument("--image", required=True, help="photo of the checkerboard")
    ap.add_argument("--Z", type=float, required=True,
                    help="tape-measured distance to the board, in mm")
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    ap.add_argument("--square-size", type=float, default=None,
                    help="true square size in mm (default: from the YAML)")
    ap.add_argument("--pattern", default=None, help="COLSxROWS (default: from YAML)")
    ap.add_argument("--tolerance", type=float, default=5.0,
                    help="max acceptable mean absolute percent error")
    ap.add_argument("--allow-scaling", action="store_true")
    args = ap.parse_args(argv)

    try:
        cam = load_params(args.params)
    except CalibrationError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2

    square = args.square_size or cam.square_size_mm
    if not square:
        print("ERROR: square size unknown; pass --square-size", file=sys.stderr)
        return 2

    if args.pattern:
        cols, rows = (int(x) for x in args.pattern.lower().split("x"))
        pattern_size = (cols, rows)
    elif cam.pattern_size:
        pattern_size = cam.pattern_size
    else:
        pattern_size = (9, 6)

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        print("ERROR: could not read %s" % args.image, file=sys.stderr)
        return 2
    h, w = img.shape[:2]

    try:
        cam, warning = check_resolution(cam, (w, h), allow_scaling=args.allow_scaling)
    except CalibrationError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2
    if warning:
        print("WARNING: %s\n" % warning)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size, FIND_FLAGS)
    if not found:
        print("ERROR: could not find a %dx%d pattern in %s"
              % (pattern_size[0], pattern_size[1], args.image), file=sys.stderr)
        return 2
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)

    print("Step 2 pipeline self-test")
    print("image        : %s (%dx%d)" % (args.image, w, h))
    print("params       : %s" % args.params)
    print("pattern      : %dx%d   true square size: %.3f mm"
          % (pattern_size[0], pattern_size[1], square))
    print("Z (measured) : %.1f mm\n" % args.Z)

    by_axis = {"horizontal": [], "vertical": []}
    for p1, p2, axis in adjacent_gaps(corners, pattern_size):
        r = measure(p1, p2, args.Z, cam)
        by_axis[axis].append(r["length_mm"])

    all_vals = np.array(by_axis["horizontal"] + by_axis["vertical"])
    print("  %-12s %5s %10s %10s %10s %10s"
          % ("axis", "n", "mean mm", "std mm", "mean err", "mean |%|"))
    for axis in ("horizontal", "vertical"):
        v = np.array(by_axis[axis])
        err = v - square
        print("  %-12s %5d %10.3f %10.3f %+10.3f %10.2f"
              % (axis, len(v), v.mean(), v.std(ddof=1), err.mean(),
                 np.mean(np.abs(err / square)) * 100))

    err = all_vals - square
    mape = float(np.mean(np.abs(err / square)) * 100)
    print("  %-12s %5d %10.3f %10.3f %+10.3f %10.2f"
          % ("combined", len(all_vals), all_vals.mean(), all_vals.std(ddof=1),
             err.mean(), mape))

    print("\n" + "-" * 62)
    print("true square size     : %.3f mm" % square)
    print("measured (mean)      : %.3f mm" % all_vals.mean())
    print("mean signed error    : %+.3f mm  (%+.2f %%)"
          % (err.mean(), err.mean() / square * 100))
    print("mean abs pct error   : %.2f %%" % mape)
    print("RMSE                 : %.3f mm" % float(np.sqrt(np.mean(err ** 2))))
    print("-" * 62)

    if mape <= args.tolerance:
        print("PASS -- within %.1f%%. Pipeline validated; proceed to Step 3."
              % args.tolerance)
        return 0

    print("FAIL -- %.2f%% exceeds the %.1f%% tolerance." % (mape, args.tolerance))
    print("Check, in order:")
    print("  1. resolution mismatch between this photo and calibration")
    print("  2. Z measured to the optical centre, not the phone's back glass")
    print("  3. the board fronto-parallel and centred in frame")
    print("  4. SQUARE_SIZE_MM in calibrate.py matching the printed board")
    return 1


if __name__ == "__main__":
    sys.exit(main())
