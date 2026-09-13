"""
Tooling -- stand in for the 20 manual click-measurements, using the synthetic
scenes, so Step 3's statistics pipeline can be exercised end to end.

    #####################################################################
    #  THIS IS NOT SUBMISSION DATA.                                     #
    #  The assignment requires 20 real measurements of real objects at a #
    #  real tape-measured distance. This script only proves analyze.py   #
    #  and geometry.py agree on a dataset whose truth is known exactly.  #
    #  Replace validation/data/measurements.csv with your own rows.      #
    #####################################################################

HOW TO RUN
    # from module2/
    python tools/simulate_measurements.py

    # options
    --truth      validation/data/synthetic_truth.csv   (from make_synthetic_data)
    --out        validation/data/measurements.csv      (what analyze.py reads)
    --params     calibration/output/camera_params.yaml
    --perfect    no click noise and no Z bias -- isolates pure code error

WHAT IT SIMULATES
    Two error sources a real capture always has, and nothing else:

    1. Click precision. The true endpoint pixels are perturbed by Gaussian
       noise (sigma = 1.5 px per point), because a human cannot click the
       exact corner.
    2. The Z reference origin. A tape measure starts at the phone's back glass,
       not the optical centre inside the lens stack, so the RECORDED Z is
       biased by -8 mm, plus 3 mm of random reading error.

    The -8 mm bias is deliberate: it is the systematic error the plan predicts
    in section 10. Note what happens with both sources active, because it is
    the real lesson: at sigma_click = 1.5 px the click term alone is about
    1% on a 155 mm object, so it SWAMPS the ~0.3% Z bias and analyze.py
    correctly reports no statistically significant bias at n = 20.

    To see the bias on its own, turn the click noise off:

        python tools/simulate_measurements.py --sigma-click 0 \
            --out validation/data/measurements_bias_only.csv
        python validation/analyze.py --csv validation/data/measurements_bias_only.csv

    Now the mean signed error is unambiguously negative and its 95% CI
    excludes zero. Running both variants is what lets the report attribute
    the error budget rather than just quote it.
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "measurement"))
from geometry import (CalibrationError, annotate, check_resolution,  # noqa: E402
                      load_params, measure)

DEFAULT_TRUTH = "validation/data/synthetic_truth.csv"
DEFAULT_OUT = "validation/data/measurements.csv"
DEFAULT_PARAMS = "calibration/output/camera_params.yaml"
DEFAULT_IMAGE_DIR = "validation/data/images"
DEFAULT_ANNOT_DIR = "validation/output/annotated"

SIGMA_CLICK_PX = 1.5
Z_BIAS_MM = -8.0
SIGMA_Z_MM = 3.0

# Schema from plan section 15. Pixel coordinates are stored so any row can be
# recomputed later if a bug in geometry.py turns up -- without reshooting.
FIELDS = ["id", "image_file", "object", "dimension", "Z_mm",
          "ground_truth_mm", "measured_mm", "u1", "v1", "u2", "v2"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Simulate the 20 Step 3 measurements")
    ap.add_argument("--truth", default=DEFAULT_TRUTH)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    ap.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    ap.add_argument("--annot-dir", default=DEFAULT_ANNOT_DIR)
    ap.add_argument("--perfect", action="store_true",
                    help="no click noise, no Z bias: isolates pure code error")
    ap.add_argument("--sigma-click", type=float, default=SIGMA_CLICK_PX,
                    help="per-point click noise in px (0 to isolate the Z bias)")
    ap.add_argument("--z-bias", type=float, default=Z_BIAS_MM,
                    help="systematic Z offset in mm (the optical-centre problem)")
    ap.add_argument("--sigma-z", type=float, default=SIGMA_Z_MM,
                    help="random tape-reading error in mm")
    ap.add_argument("--seed", type=int, default=20260913)
    args = ap.parse_args(argv)

    if not os.path.exists(args.truth):
        print("ERROR: %s not found. Run first:\n"
              "  python tools/make_synthetic_data.py --only measure" % args.truth,
              file=sys.stderr)
        return 2
    try:
        cam = load_params(args.params)
    except CalibrationError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    sigma_click = 0.0 if args.perfect else args.sigma_click
    z_bias = 0.0 if args.perfect else args.z_bias
    sigma_z = 0.0 if args.perfect else args.sigma_z

    print("=" * 78)
    print("SIMULATED STEP 3 MEASUREMENTS -- synthetic, not submission data")
    print("=" * 78)
    print("params      : %s  (fx=%.2f @ %dx%d)"
          % (args.params, cam.fx, cam.width, cam.height))
    print("click noise : sigma = %.2f px per point" % sigma_click)
    print("Z recording : bias %+.1f mm, sigma %.1f mm\n" % (z_bias, sigma_z))

    os.makedirs(args.annot_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    with open(args.truth, newline="") as fh:
        truth_rows = list(csv.DictReader(fh))

    out_rows = []
    print("  %-3s %-18s %-3s %8s %10s %10s %9s"
          % ("id", "object", "dim", "Z rec", "truth mm", "meas mm", "err mm"))

    for row in truth_rows:
        img_path = os.path.join(args.image_dir, row["image_file"])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print("  [SKIP] %s unreadable" % row["image_file"])
            continue
        h, w = img.shape[:2]
        cam_i, _ = check_resolution(cam, (w, h))

        # Where a perfect click would land, then human click noise.
        u1 = float(row["u1_true"]) + rng.normal(0, sigma_click)
        v1 = float(row["v1_true"]) + rng.normal(0, sigma_click)
        u2 = float(row["u2_true"]) + rng.normal(0, sigma_click)
        v2 = float(row["v2_true"]) + rng.normal(0, sigma_click)

        # What the tape measure reads, not the true optical-centre distance.
        z_true = float(row["Z_true_mm"])
        z_recorded = z_true + z_bias + rng.normal(0, sigma_z)

        result = measure((u1, v1), (u2, v2), z_recorded, cam_i)
        gt = float(row["ground_truth_mm"])
        measured = result["length_mm"]

        cv2.imwrite(
            os.path.join(args.annot_dir,
                         os.path.splitext(row["image_file"])[0] + "_annotated.jpg"),
            annotate(img, (u1, v1), (u2, v2), result,
                     label="%s %s: %.1f mm (true %.1f)"
                           % (row["object"], row["dimension"], measured, gt)),
            [cv2.IMWRITE_JPEG_QUALITY, 88])

        out_rows.append({
            "id": row["id"], "image_file": row["image_file"],
            "object": row["object"], "dimension": row["dimension"],
            "Z_mm": "%.1f" % z_recorded,
            "ground_truth_mm": "%.2f" % gt,
            "measured_mm": "%.2f" % measured,
            "u1": "%.2f" % u1, "v1": "%.2f" % v1,
            "u2": "%.2f" % u2, "v2": "%.2f" % v2,
        })
        print("  %-3s %-18s %-3s %8.1f %10.1f %10.1f %+9.2f"
              % (row["id"], row["object"], row["dimension"],
                 z_recorded, gt, measured, measured - gt))

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    print("\nwrote %s  (%d rows)" % (args.out, len(out_rows)))
    print("wrote annotated images to %s/" % args.annot_dir)
    print("\nnext:  python validation/analyze.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
