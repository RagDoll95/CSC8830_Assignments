"""
Tooling -- one command that proves the whole pipeline is correct.

WHY IT EXISTS
    Running the scripts shows they do not crash. This scores them. Because the
    synthetic dataset's ground-truth K, distortion and object sizes are known
    exactly, every stage can be checked against a number rather than eyeballed:

        Step 1  recovered f_x, c_x, k1 vs. the true values
        Step 2  geometry unit tests, then the 25 mm board measured at a known Z
        Step 3  pure-code error with all capture noise switched off
        guards  resolution mismatch refused, Z <= 0 refused, CLI == web app

    If this passes and real measurements are still off, the problem is in the
    capture (Z origin, resolution, focus), not in the code.

HOW TO RUN
    # from module2/
    python tools/verify_pipeline.py            # assumes synthetic data exists
    python tools/verify_pipeline.py --regenerate   # rebuild the data first

EXIT CODE
    0 if every check passes, 1 otherwise.
"""

import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # week_2/module2
REPO_ROOT = os.path.dirname(os.path.dirname(ROOT))  # the web app lives here
sys.path.insert(0, os.path.join(ROOT, "measurement"))
sys.path.insert(0, HERE)

from geometry import (CalibrationError, check_resolution, load_params,  # noqa: E402
                      measure)
import make_synthetic_data as synth  # noqa: E402

PARAMS = os.path.join(ROOT, "calibration", "output", "camera_params.yaml")
SELFTEST_IMAGE = os.path.join(ROOT, "measurement", "data",
                              "synthetic_selftest_board.jpg")
SELFTEST_Z = 1000.0

PY = sys.executable
results = []


def check(label, ok, detail=""):
    results.append((label, bool(ok), detail))
    print("  [%s] %-46s %s" % ("PASS" if ok else "FAIL", label, detail))
    return ok


def run(cmd, label):
    """Run a pipeline script and check it exits 0."""
    proc = subprocess.run([PY] + cmd, cwd=ROOT, capture_output=True, text=True)
    ok = proc.returncode == 0
    if not ok:
        print((proc.stdout or "")[-1500:])
        print((proc.stderr or "")[-1500:], file=sys.stderr)
    check(label, ok, "exit %d" % proc.returncode)
    return proc


def main(argv=None):
    ap = argparse.ArgumentParser(description="Verify the whole pipeline")
    ap.add_argument("--regenerate", action="store_true",
                    help="regenerate synthetic data and recalibrate first")
    args = ap.parse_args(argv)

    print("=" * 74)
    print("PIPELINE VERIFICATION against known synthetic ground truth")
    print("=" * 74)

    if args.regenerate:
        print("\n[0] regenerating synthetic data")
        run(["tools/make_synthetic_data.py"], "make_synthetic_data.py runs")
        run(["calibration/calibrate.py", "--notes",
             "SYNTHETIC dataset -- pipeline verification only"],
            "calibrate.py runs")
        run(["tools/simulate_measurements.py"], "simulate_measurements.py runs")

    if not os.path.exists(PARAMS):
        print("\nERROR: %s missing. Run with --regenerate." % PARAMS,
              file=sys.stderr)
        return 1

    # ---------------------------------------------------------------- Step 1
    print("\n[1] STEP 1 -- calibration vs. known ground truth")
    cam = load_params(PARAMS)
    truth = {
        "f_x": (cam.fx, synth.FX_TRUE),
        "f_y": (cam.fy, synth.FY_TRUE),
        "c_x": (cam.cx, synth.CX_TRUE),
        "c_y": (cam.cy, synth.CY_TRUE),
    }
    for name, (got, want) in truth.items():
        rel = abs(got - want) / abs(want) * 100
        check("%s within 0.5%% of truth" % name, rel < 0.5,
              "%.3f vs %.3f  (%.4f%%)" % (got, want, rel))

    k1_got = float(cam.dist.ravel()[0])
    k1_true = float(synth.DIST_TRUE.ravel()[0])
    check("k1 within 5% of truth", abs(k1_got - k1_true) / abs(k1_true) < 0.05,
          "%.5f vs %.5f" % (k1_got, k1_true))
    check("RMS reprojection error < 0.5 px", (cam.rms or 9) < 0.5,
          "%.4f px" % (cam.rms or -1))
    check("image count >= 15", (cam.image_count or 0) >= 15,
          "%d images" % (cam.image_count or 0))
    check("resolution matches truth",
          cam.image_size == tuple(synth.IMAGE_SIZE),
          "%dx%d" % cam.image_size)

    # ---------------------------------------------------------------- Step 2
    print("\n[2] STEP 2 -- geometry")
    proc = subprocess.run([PY, "measurement/geometry.py"], cwd=ROOT,
                          capture_output=True, text=True)
    check("geometry.py self-test passes", proc.returncode == 0,
          "ALL CHECKS PASSED" if proc.returncode == 0 else "see output")

    # Analytic round trip: project a known size, measure it back.
    Z, W_true = 2500.0, 300.0
    u1 = cam.fx * (-W_true / 2) / Z + cam.cx
    u2 = cam.fx * (W_true / 2) / Z + cam.cx
    r = measure((u1, cam.cy), (u2, cam.cy), Z, cam, undistort=False)
    check("pinhole round trip exact", abs(r["width_mm"] - W_true) < 1e-6,
          "%.6f mm vs %.1f" % (r["width_mm"], W_true))

    # The real test: measure the 25 mm board from a rendered image.
    print("\n[3] STEP 2 -- checkerboard self-test on a rendered image")
    if os.path.exists(SELFTEST_IMAGE):
        img = cv2.imread(SELFTEST_IMAGE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pattern = cam.pattern_size or synth.PATTERN_SIZE
        found, corners = cv2.findChessboardCorners(
            gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if check("board detected in the self-test image", found):
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            cols, rows = pattern
            grid = corners.reshape(rows, cols, 2)
            gaps = []
            for rr in range(rows):
                for cc in range(cols - 1):
                    gaps.append(measure(grid[rr, cc], grid[rr, cc + 1],
                                        SELFTEST_Z, cam)["length_mm"])
            gaps = np.array(gaps)
            mape = float(np.mean(np.abs(gaps - synth.SQUARE_MM)) / synth.SQUARE_MM * 100)
            check("25 mm square measured within 1%", mape < 1.0,
                  "mean %.3f mm, MAPE %.3f%%" % (gaps.mean(), mape))
    else:
        check("self-test image present", False, SELFTEST_IMAGE)

    # ---------------------------------------------------------------- Step 3
    print("\n[4] STEP 3 -- pure code error, all capture noise off")
    tmp_csv = os.path.join(ROOT, "validation", "output", "_verify_perfect.csv")
    proc = subprocess.run(
        [PY, "tools/simulate_measurements.py", "--perfect",
         "--out", os.path.relpath(tmp_csv, ROOT),
         "--annot-dir", "validation/output/_verify_annot"],
        cwd=ROOT, capture_output=True, text=True)
    if check("perfect-capture simulation runs", proc.returncode == 0):
        import csv as _csv
        with open(tmp_csv, newline="") as fh:
            rows = list(_csv.DictReader(fh))
        gt = np.array([float(r["ground_truth_mm"]) for r in rows])
        me = np.array([float(r["measured_mm"]) for r in rows])
        mape = float(np.mean(np.abs((me - gt) / gt)) * 100)
        check("20 rows produced", len(rows) == 20, "%d rows" % len(rows))
        check("pure code error < 0.1%", mape < 0.1, "MAPE %.4f%%" % mape)

    proc = run(["validation/analyze.py", "--csv",
                os.path.relpath(tmp_csv, ROOT), "--out-dir",
                "validation/output/_verify_out", "--no-plots"],
               "analyze.py runs")

    # Recompute-from-pixels must reproduce the stored values.
    proc = subprocess.run(
        [PY, "validation/analyze.py", "--csv", os.path.relpath(tmp_csv, ROOT),
         "--recompute", "--out-dir", "validation/output/_verify_out",
         "--no-plots"], cwd=ROOT, capture_output=True, text=True)
    check("analyze.py --recompute reproduces the CSV", proc.returncode == 0)

    # ---------------------------------------------------------------- guards
    print("\n[5] GUARDS")
    try:
        check_resolution(cam, (1920, 1080))
        check("resolution mismatch refused", False, "no exception raised")
    except CalibrationError:
        check("resolution mismatch refused", True, "CalibrationError")

    scaled, warn = check_resolution(cam, (cam.width // 2, cam.height // 2),
                                    allow_scaling=True)
    check("allow_scaling halves f_x and warns",
          abs(scaled.fx - cam.fx / 2) < 1e-6 and warn is not None,
          "f_x %.2f -> %.2f" % (cam.fx, scaled.fx))

    for label, args_ in (("Z = 0 refused", (0.0,)), ("Z < 0 refused", (-1.0,))):
        try:
            measure((10, 10), (200, 200), args_[0], cam)
            check(label, False, "no exception raised")
        except ValueError:
            check(label, True, "ValueError")
    try:
        measure((5, 5), (5, 5), 1000.0, cam)
        check("identical points refused", False, "no exception raised")
    except ValueError:
        check("identical points refused", True, "ValueError")

    # ------------------------------------------------------ CLI == web app
    print("\n[6] CLI and WEB APP agree (the section-25 scaling gotcha)")
    img_dir = os.path.join(ROOT, "validation", "data", "images")
    sample = os.path.join(img_dir, "synthetic_meas_03.jpg")
    if os.path.exists(sample):
        p1, p2, Zs = (400.0, 500.0), (1500.0, 900.0), 2320.0
        proc = subprocess.run(
            [PY, "measurement/measure_cli.py", "--image",
             os.path.relpath(sample, ROOT), "--Z", str(Zs),
             "--points", "%g,%g" % p1, "%g,%g" % p2, "--json",
             "--out", "validation/output/_verify_cli.jpg"],
            cwd=ROOT, capture_output=True, text=True)
        if check("measure_cli.py --points runs", proc.returncode == 0):
            import json as _json
            cli = _json.loads(proc.stdout)["result"]["length_mm"]
            direct = measure(p1, p2, Zs, cam)["length_mm"]
            check("CLI matches geometry.measure directly",
                  abs(cli - direct) < 1e-6, "%.6f vs %.6f mm" % (cli, direct))

            # The web app must scale display clicks back to full resolution
            # before the geometry ever sees them.
            # The web layer moved up to <repo>/app, where one application
            # serves every assignment; Module 2's pages are a blueprint in it.
            sys.path.insert(0, os.path.join(REPO_ROOT, "app"))
            try:
                from modules.module2 import scale_click
                disp_w = 800
                img = cv2.imread(sample)
                nat_w, nat_h = img.shape[1], img.shape[0]
                disp_h = int(round(nat_h * disp_w / nat_w))
                s1 = scale_click(p1[0] * disp_w / nat_w, p1[1] * disp_h / nat_h,
                                 disp_w, disp_h, nat_w, nat_h)
                s2 = scale_click(p2[0] * disp_w / nat_w, p2[1] * disp_h / nat_h,
                                 disp_w, disp_h, nat_w, nat_h)
                web = measure(s1, s2, Zs, cam)["length_mm"]
                check("web-app click scaling matches the CLI",
                      abs(web - direct) / direct < 0.005,
                      "%.3f vs %.3f mm" % (web, direct))
            except ImportError as exc:
                check("web-app click scaling matches the CLI", False, str(exc))
    else:
        check("measurement sample image present", False, sample)

    # ---------------------------------------------------------------- summary
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print("\n" + "=" * 74)
    print("%d / %d checks passed" % (passed, total))
    if passed == total:
        print("PIPELINE VERIFIED. Any remaining error in real measurements is a")
        print("capture problem (Z origin, resolution, focus), not a code problem.")
    else:
        print("FAILURES:")
        for label, ok, detail in results:
            if not ok:
                print("  - %s  %s" % (label, detail))
    print("=" * 74)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
