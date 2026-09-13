"""
Step 3 -- error statistics over the 20 validation measurements.

WHAT IT DOES
    Reads one CSV of measurements and reports the statistics the assignment
    asks for: signed error, mean signed error (systematic bias), standard
    deviation (random spread), MAE, RMSE, percent error and MAPE, plus the
    min/max error and which measurement produced each. Also writes the two
    diagnostic plots -- error vs. ground-truth size and error vs. Z.

    Kept separate from the measuring code on purpose: the numbers in the
    report are reproducible by re-running this one command on the CSV.

HOW TO RUN
    # from module2/
    python validation/analyze.py
    python validation/analyze.py --csv validation/data/measurements.csv
    python validation/analyze.py --no-plots        # skip matplotlib

INPUT CSV SCHEMA (plan section 15)
    id, image_file, object, dimension, Z_mm,
    ground_truth_mm, measured_mm, u1, v1, u2, v2

    Storing u1..v2 means any row can be recomputed if geometry.py changes,
    without reshooting. --recompute does exactly that.

OUTPUTS
    validation/output/statistics.txt        the printed report
    validation/output/error_vs_size.png     error vs. ground-truth size
    validation/output/error_vs_Z.png        error vs. distance
    validation/output/measurements_with_errors.csv

HOW TO READ IT
    A nonzero MEAN SIGNED ERROR is not noise -- it is bias, and the usual
    cause is the Z reference origin (tape measured from the phone's back
    glass rather than the optical centre inside the lens stack) or a residual
    focal-length error. The STD is the random part, and it should track the
    error-propagation prediction in geometry.measure_uncertainty.
"""

import argparse
import csv
import io
import os
import sys

import numpy as np

DEFAULT_CSV = "validation/data/measurements.csv"
DEFAULT_OUT_DIR = "validation/output"
REQUIRED = ["id", "image_file", "object", "dimension", "Z_mm",
            "ground_truth_mm", "measured_mm"]


def read_csv(path):
    if not os.path.exists(path):
        raise SystemExit(
            "ERROR: %s not found.\n"
            "Log your 20 measurements there (schema in this file's docstring),\n"
            "or generate the synthetic stand-in:\n"
            "  python tools/make_synthetic_data.py --only measure\n"
            "  python tools/simulate_measurements.py" % path)

    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("ERROR: %s has no data rows." % path)

    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        raise SystemExit("ERROR: %s is missing column(s): %s"
                         % (path, ", ".join(missing)))
    return rows


def recompute(rows, params_path):
    """Re-derive measured_mm from the stored pixels and the current geometry.py.

    This is why the pixel coordinates are in the schema: a bug fix in
    geometry.py does not mean reshooting 20 photos.
    """
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "measurement"))
    from geometry import load_params, measure

    cam = load_params(params_path)
    for r in rows:
        for key in ("u1", "v1", "u2", "v2"):
            if key not in r or r[key] in (None, ""):
                raise SystemExit("ERROR: --recompute needs u1,v1,u2,v2 in the CSV")
        res = measure((float(r["u1"]), float(r["v1"])),
                      (float(r["u2"]), float(r["v2"])),
                      float(r["Z_mm"]), cam)
        r["measured_mm"] = "%.4f" % res["length_mm"]
    return rows


def compute(rows):
    gt = np.array([float(r["ground_truth_mm"]) for r in rows])
    meas = np.array([float(r["measured_mm"]) for r in rows])
    Z = np.array([float(r["Z_mm"]) for r in rows])

    err = meas - gt                      # signed error, mm
    pct = err / gt * 100.0               # signed percent error

    stats = {
        "n": len(rows),
        "mean_signed_error_mm": float(err.mean()),
        "std_error_mm": float(err.std(ddof=1)) if len(err) > 1 else 0.0,
        "mae_mm": float(np.abs(err).mean()),
        "rmse_mm": float(np.sqrt(np.mean(err ** 2))),
        "mean_signed_pct": float(pct.mean()),
        "mape_pct": float(np.abs(pct).mean()),
        "std_pct": float(pct.std(ddof=1)) if len(pct) > 1 else 0.0,
        "min_error_mm": float(err.min()),
        "max_error_mm": float(err.max()),
        "min_error_idx": int(np.argmin(err)),
        "max_error_idx": int(np.argmax(err)),
        "max_abs_error_idx": int(np.argmax(np.abs(err))),
        "Z_min_mm": float(Z.min()),
        "Z_max_mm": float(Z.max()),
        "gt_min_mm": float(gt.min()),
        "gt_max_mm": float(gt.max()),
    }
    # 95% CI on the mean -- says whether the bias is real or just spread.
    if len(err) > 1:
        sem = stats["std_error_mm"] / np.sqrt(stats["n"])
        stats["sem_mm"] = float(sem)
        stats["ci95_mm"] = (float(err.mean() - 1.96 * sem),
                            float(err.mean() + 1.96 * sem))
        stats["bias_significant"] = not (stats["ci95_mm"][0] <= 0 <= stats["ci95_mm"][1])
    # Correlations: a trend points at a cause rather than noise.
    if len(err) > 2:
        stats["corr_err_vs_gt"] = float(np.corrcoef(gt, err)[0, 1])
        stats["corr_pct_vs_Z"] = float(np.corrcoef(Z, pct)[0, 1])
    return gt, meas, Z, err, pct, stats


def report(rows, gt, meas, Z, err, pct, stats):
    out = io.StringIO()
    w = out.write

    w("=" * 96 + "\n")
    w("STEP 3 -- VALIDATION OVER %d MEASUREMENTS\n" % stats["n"])
    w("=" * 96 + "\n\n")

    w("  %-3s %-18s %-4s %9s %11s %11s %9s %8s\n"
      % ("id", "object", "dim", "Z (mm)", "truth (mm)", "meas (mm)",
         "err (mm)", "err (%)"))
    w("  " + "-" * 92 + "\n")
    for i, r in enumerate(rows):
        w("  %-3s %-18s %-4s %9.1f %11.2f %11.2f %+9.2f %+8.2f\n"
          % (r["id"], r["object"][:18], r["dimension"], Z[i], gt[i], meas[i],
             err[i], pct[i]))
    w("\n")

    w("-" * 96 + "\n")
    w("ERROR STATISTICS\n")
    w("-" * 96 + "\n")
    w("  measurements                n     : %d\n" % stats["n"])
    w("  ground-truth size range           : %.1f .. %.1f mm\n"
      % (stats["gt_min_mm"], stats["gt_max_mm"]))
    w("  distance range               Z    : %.0f .. %.0f mm\n"
      % (stats["Z_min_mm"], stats["Z_max_mm"]))
    w("\n")
    w("  mean signed error                 : %+8.3f mm   <- systematic bias\n"
      % stats["mean_signed_error_mm"])
    w("  std deviation of error            : %8.3f mm   <- random spread\n"
      % stats["std_error_mm"])
    w("  mean absolute error         MAE   : %8.3f mm\n" % stats["mae_mm"])
    w("  root-mean-square error      RMSE  : %8.3f mm\n" % stats["rmse_mm"])
    w("\n")
    w("  mean signed percent error         : %+8.3f %%\n" % stats["mean_signed_pct"])
    w("  mean absolute percent error MAPE  : %8.3f %%\n" % stats["mape_pct"])
    w("  std deviation of percent error    : %8.3f %%\n" % stats["std_pct"])
    w("\n")
    lo = rows[stats["min_error_idx"]]
    hi = rows[stats["max_error_idx"]]
    worst = rows[stats["max_abs_error_idx"]]
    w("  most negative error               : %+8.3f mm  (id %s, %s %s)\n"
      % (stats["min_error_mm"], lo["id"], lo["object"], lo["dimension"]))
    w("  most positive error               : %+8.3f mm  (id %s, %s %s)\n"
      % (stats["max_error_mm"], hi["id"], hi["object"], hi["dimension"]))
    w("  largest absolute error            : %+8.3f mm  (id %s, %s %s)\n"
      % (err[stats["max_abs_error_idx"]], worst["id"], worst["object"],
         worst["dimension"]))

    if "ci95_mm" in stats:
        w("\n")
        w("  standard error of the mean        : %8.3f mm\n" % stats["sem_mm"])
        w("  95%% CI on the mean signed error   : [%+.3f, %+.3f] mm\n"
          % stats["ci95_mm"])
        if stats["bias_significant"]:
            w("  -> the CI excludes zero: the bias is REAL, not sampling noise.\n")
            w("     Attribute it. The leading candidate is the Z reference origin\n")
            w("     (a tape measure starts at the back glass, while Z is defined\n")
            w("     from the optical centre inside the lens stack); a residual\n")
            w("     focal-length error from Step 1 is the other candidate.\n")
        else:
            w("  -> the CI contains zero: no statistically significant bias.\n")

    if "corr_err_vs_gt" in stats:
        w("\n")
        w("-" * 96 + "\n")
        w("TREND DIAGNOSTICS\n")
        w("-" * 96 + "\n")
        w("  corr(ground-truth size, signed error)  : %+.3f\n" % stats["corr_err_vs_gt"])
        w("     A strong positive value means error grows in proportion to size,\n")
        w("     i.e. a SCALE error -- focal length or Z, not click precision.\n")
        w("  corr(Z, percent error)                 : %+.3f\n" % stats["corr_pct_vs_Z"])
        w("     A trend here implicates the Z measurement itself: a fixed origin\n")
        w("     offset dZ produces a percent error dZ/Z that shrinks with distance.\n")

    w("\n")
    w("-" * 96 + "\n")
    w("PER-DIMENSION BREAKDOWN\n")
    w("-" * 96 + "\n")
    w("  %-12s %5s %12s %12s %10s\n"
      % ("dimension", "n", "mean err mm", "std err mm", "MAPE %"))
    for dim in sorted(set(r["dimension"] for r in rows)):
        mask = np.array([r["dimension"] == dim for r in rows])
        e, p = err[mask], pct[mask]
        w("  %-12s %5d %+12.3f %12.3f %10.3f\n"
          % (dim, mask.sum(), e.mean(),
             e.std(ddof=1) if mask.sum() > 1 else 0.0, np.abs(p).mean()))

    w("\n")
    w("-" * 96 + "\n")
    w("ERROR PROPAGATION CHECK (plan section 12)\n")
    w("-" * 96 + "\n")
    w("  W = du * Z / f_x, so relative errors add:\n")
    w("      dW/W  ~=  d(du)/du  +  dZ/Z  +  df_x/f_x\n")
    w("  Observed spread of percent error : %.3f %%\n" % stats["std_pct"])
    w("  The click-precision term shrinks as the pixel span grows, which is why\n")
    w("  larger objects measure proportionally better -- visible in the size\n")
    w("  breakdown above and in error_vs_size.png.\n")
    w("=" * 96 + "\n")
    return out.getvalue()


def make_plots(gt, Z, err, pct, rows, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots "
              "(pip install matplotlib)")
        return []

    written = []
    dims = np.array([r["dimension"] for r in rows])
    colours = {"W": "#1f77b4", "H": "#d62728", "D": "#2ca02c"}

    # Error vs. ground-truth size.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for dim in sorted(set(dims)):
        m = dims == dim
        ax1.scatter(gt[m], err[m], label=dim, s=46,
                    color=colours.get(dim, "#777777"), edgecolor="k", linewidth=0.5)
        ax2.scatter(gt[m], pct[m], label=dim, s=46,
                    color=colours.get(dim, "#777777"), edgecolor="k", linewidth=0.5)
    for ax, ylabel, title in ((ax1, "signed error (mm)", "Absolute error vs. size"),
                              (ax2, "signed error (%)", "Percent error vs. size")):
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("ground-truth dimension (mm)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(title="dim", fontsize=8)
    if len(gt) > 2:
        fit = np.polyfit(gt, err, 1)
        xs = np.linspace(gt.min(), gt.max(), 50)
        ax1.plot(xs, np.polyval(fit, xs), color="#555555", lw=1.2,
                 label="fit: %+.4f mm/mm" % fit[0])
        ax1.legend(fontsize=8)
    fig.suptitle("Step 3: error vs. ground-truth dimension (n=%d)" % len(gt))
    fig.tight_layout()
    path = os.path.join(out_dir, "error_vs_size.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    # Error vs. Z.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for dim in sorted(set(dims)):
        m = dims == dim
        ax1.scatter(Z[m], err[m], label=dim, s=46,
                    color=colours.get(dim, "#777777"), edgecolor="k", linewidth=0.5)
        ax2.scatter(Z[m], pct[m], label=dim, s=46,
                    color=colours.get(dim, "#777777"), edgecolor="k", linewidth=0.5)
    for ax, ylabel, title in ((ax1, "signed error (mm)", "Absolute error vs. distance"),
                              (ax2, "signed error (%)", "Percent error vs. distance")):
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("Z, object distance (mm)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(title="dim", fontsize=8)
    fig.suptitle("Step 3: error vs. distance (n=%d)" % len(Z))
    fig.tight_layout()
    path = os.path.join(out_dir, "error_vs_Z.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    # Error distribution, with the mean marked -- makes bias visually obvious.
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.hist(err, bins=max(5, min(12, len(err) // 2)), color="#6699cc",
            edgecolor="k", alpha=0.85)
    ax.axvline(0, color="k", lw=1.0, ls="--", label="zero error")
    ax.axvline(err.mean(), color="#d62728", lw=1.6,
               label="mean %+.2f mm" % err.mean())
    ax.set_xlabel("signed error (mm)")
    ax.set_ylabel("count")
    ax.set_title("Error distribution (n=%d)" % len(err))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(out_dir, "error_distribution.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)
    return written


def write_augmented_csv(rows, err, pct, path):
    fields = list(rows[0].keys())
    for extra in ("error_mm", "abs_error_mm", "percent_error"):
        if extra not in fields:
            fields.append(extra)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i, r in enumerate(rows):
            out = dict(r)
            out["error_mm"] = "%.4f" % err[i]
            out["abs_error_mm"] = "%.4f" % abs(err[i])
            out["percent_error"] = "%.4f" % pct[i]
            writer.writerow(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 3: error statistics")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--recompute", action="store_true",
                    help="re-derive measured_mm from the stored pixels")
    ap.add_argument("--params", default="calibration/output/camera_params.yaml")
    args = ap.parse_args(argv)

    rows = read_csv(args.csv)
    if args.recompute:
        print("recomputing measured_mm from stored pixel coordinates...\n")
        rows = recompute(rows, args.params)

    if len(rows) < 20:
        print("NOTE: only %d rows; the assignment asks for 20 measurements.\n"
              % len(rows))

    gt, meas, Z, err, pct, stats = compute(rows)
    text = report(rows, gt, meas, Z, err, pct, stats)
    print(text)

    os.makedirs(args.out_dir, exist_ok=True)
    stats_path = os.path.join(args.out_dir, "statistics.txt")
    with open(stats_path, "w") as fh:
        fh.write(text)
    print("wrote %s" % stats_path)

    aug = os.path.join(args.out_dir, "measurements_with_errors.csv")
    write_augmented_csv(rows, err, pct, aug)
    print("wrote %s" % aug)

    if not args.no_plots:
        for path in make_plots(gt, Z, err, pct, rows, args.out_dir):
            print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
