"""
Step 1 (pre-flight) -- sanity-check calibration photos BEFORE fitting.

WHY
    Refitting is cheap; reshooting is not. This reports, per image: whether the
    9x6 corners are detectable, resolution consistency, blur (variance of
    Laplacian), and where in the frame the board sits -- so you can tell at a
    glance whether the shot list actually covers the frame corners, which is
    where distortion lives.

HOW TO RUN
    # from module2/
    python calibration/capture_check.py
    python calibration/capture_check.py --images "calibration/data/images/*.jpg"

READ THE OUTPUT AS
    detected  : must be >= 15 for the plan's exit criteria
    blur      : variance of Laplacian; < 100 is suspect on a phone photo
    coverage  : fraction of an 3x3 frame grid that at least one board touches.
                Low coverage => reshoot with the board nearer the frame edges.
"""

import argparse
import os
import sys

import cv2
import numpy as np

from calibrate import FIND_FLAGS, IMAGE_GLOB, PATTERN_SIZE, list_images, parse_pattern

BLUR_WARN = 100.0  # variance-of-Laplacian threshold


def check(paths, pattern_size):
    rows = []
    sizes = {}
    grid_hits = np.zeros((3, 3), dtype=int)

    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        name = os.path.basename(path)
        if img is None:
            rows.append((name, "-", "unreadable", 0.0, "-"))
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        sizes.setdefault((w, h), []).append(name)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        found, corners = cv2.findChessboardCorners(gray, pattern_size, FIND_FLAGS)
        if not found:
            rows.append((name, "%dx%d" % (w, h), "NOT FOUND", blur, "-"))
            continue

        pts = corners.reshape(-1, 2)
        cxy = pts.mean(axis=0)
        # Which cell of a 3x3 frame grid the board centroid lands in.
        gx = min(2, int(cxy[0] / w * 3))
        gy = min(2, int(cxy[1] / h * 3))
        grid_hits[gy, gx] += 1
        # Board extent as a fraction of frame area -- proxy for distance.
        span = (np.ptp(pts[:, 0]) / w) * (np.ptp(pts[:, 1]) / h)
        rows.append((name, "%dx%d" % (w, h), "ok", blur,
                     "cell(%d,%d) area=%.2f" % (gy, gx, span)))

    return rows, sizes, grid_hits


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pre-flight check on calibration photos")
    ap.add_argument("--images", default=IMAGE_GLOB)
    ap.add_argument("--pattern", default="%dx%d" % PATTERN_SIZE)
    args = ap.parse_args(argv)

    pattern_size = parse_pattern(args.pattern)
    paths = list_images(args.images)
    if not paths:
        print("ERROR: no images matched %r" % args.images, file=sys.stderr)
        return 2

    print("checking %d images for a %dx%d pattern\n" % (len(paths), *pattern_size))
    rows, sizes, grid_hits = check(paths, pattern_size)

    print("  %-34s %-12s %-10s %8s  %s"
          % ("image", "resolution", "corners", "blur", "position"))
    for name, res, status, blur, pos in rows:
        flag = " <-- blurry?" if (status == "ok" and blur < BLUR_WARN) else ""
        print("  %-34s %-12s %-10s %8.1f  %s%s"
              % (name, res, status, blur, pos, flag))

    detected = sum(1 for r in rows if r[2] == "ok")
    coverage = float((grid_hits > 0).sum()) / 9.0

    print("\n" + "-" * 62)
    print("detected        : %d / %d" % (detected, len(paths)))
    print("frame coverage  : %.0f%% of a 3x3 grid" % (coverage * 100))
    print("board centroids per frame cell:")
    for r in range(3):
        print("   " + "  ".join("%3d" % grid_hits[r, c] for c in range(3)))

    if len(sizes) > 1:
        print("\nWARNING: mixed resolutions -- calibrate.py will reject the odd ones.")
        for size, names in sizes.items():
            print("  %dx%d : %d image(s)" % (size[0], size[1], len(names)))
    elif sizes:
        (w, h), _ = next(iter(sizes.items()))
        print("\nsingle resolution %dx%d -- good. Use this same resolution in Steps 2-3."
              % (w, h))

    print("-" * 62)
    if detected < 15:
        print("ACTION: only %d usable images; shoot more before fitting." % detected)
    if coverage < 0.6:
        print("ACTION: coverage is thin -- put the board near the frame corners too.")
    if detected >= 15 and coverage >= 0.6:
        print("Looks good. Run:  python calibration/calibrate.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
