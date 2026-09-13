# Module 2 — Camera Calibration and Real-World Measurement

**CSc 8830: Computer Vision** · Python + OpenCV + Flask

Camera calibration by Zhang's multi-plane method, real-world 2D measurement by inverting
the perspective projection at a known object distance, a 20-measurement validation
experiment with error statistics, the two-camera derivation, and a web application that
reaches all four parts from one page.

---

## ⚠️ Read this first: the data in this repo is SYNTHETIC

Every image and CSV currently committed is named `synthetic_*` and was **rendered** by
`tools/make_synthetic_data.py`, not photographed. It exists for one reason: because the
ground-truth `K`, distortion coefficients, and object sizes are known exactly, the code
can be **scored** rather than merely run.

| What the synthetic data proves | What it does not do |
|---|---|
| `calibrate.py` recovers `f_x` to within **0.013%** of the true value | Satisfy the assignment |
| The measurement pipeline is exact to **0.01%** with no capture noise | Count as 20 real measurements |
| The guards, the CLI, and the web app all agree | Substitute for your own camera's `K` |

**Before submitting you must replace it with your own captures.** The three places to
swap, and nothing else changes:

1. `calibration/data/images/` — your 15–20 checkerboard photos
2. `validation/data/images/` — your 20 measurement photos
3. `validation/data/measurements.csv` — your 20 logged rows with ruler ground truth

Then delete the synthetic files and re-run the commands in
[Running the pipeline](#running-the-pipeline).

> **→ [`NEXT_STEPS.md`](NEXT_STEPS.md) is the run sheet for doing exactly that.** It walks
> the capture protocol, the 20-measurement procedure, the analysis, the screen recording,
> and the PDF, with a troubleshooting table for when the numbers look wrong. Start there.

---

## Install

```bash
cd module2
python -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt
```

Verified on Python 3.9.23 with opencv-python 5.0.0, numpy 2.0.2, Flask 3.1.3,
matplotlib 3.9.4.

---

## Launch the web application

```bash
cd module2
python app/app.py
# open http://127.0.0.1:5000
```

Options: `--port 8000`, `--host 0.0.0.0`, `--debug`.

The nav bar reaches all four parts, so a screen recording can walk the whole assignment
without retyping URLs:

| Page | Part | What it does |
|---|---|---|
| `/` | — | Overview, loaded camera, CLI equivalents |
| `/calibration` | A · Step 1 | Upload board images → fit `K` and distortion → per-image error table → corner overlays → download `camera_params.yaml` |
| `/measurement` | B · Step 2 | Upload an image, enter `Z`, click two points → width/height/diagonal + error budget + annotated image |
| `/validation` | C · Step 3 | Measurement table, full error statistics, diagnostic plots |
| `/theory` | D | The two-camera derivation |

---

## Running the pipeline

Every script carries its own `HOW TO RUN` block in its module docstring. Run all commands
**from `module2/`**.

### Step 1 — Calibration (Part A)

```bash
# Pre-flight the photos BEFORE fitting: detection, resolution consistency,
# blur, and whether the shot list actually covers the frame corners.
python calibration/capture_check.py

# Fit. Writes camera_params.yaml, the per-image error table, and corner overlays.
python calibration/calibrate.py

# With your own board and metadata:
python calibration/calibrate.py \
    --pattern 9x6 --square-size 25.0 \
    --images "calibration/data/images/*.jpg" \
    --notes "iPhone 14 Pro, native camera, 4032x3024, AE/AF locked"
```

Outputs: `calibration/output/camera_params.yaml`,
`calibration/output/reprojection_errors.csv`, `calibration/output/debug/*.jpg`.

### Step 2 — Measurement (Part B)

```bash
# Geometry unit tests -- no photos or calibration needed.
python measurement/geometry.py

# Validate the WHOLE pipeline against the board, whose square size is known.
# Do this before burning 20 experimental measurements.
python measurement/selftest_checkerboard.py \
    --image measurement/data/synthetic_selftest_board.jpg --Z 1000

# Interactive: click two points (r = reset, s = save, q = quit).
python measurement/measure_cli.py --image PHOTO.jpg --Z 2400

# Non-interactive / scriptable.
python measurement/measure_cli.py --image PHOTO.jpg --Z 2400 \
    --points 120,340 980,352 --json

# Step 3 logging: measure and append the row to the CSV in one command, so no
# pixel coordinate is ever retyped. Interactive: press 's' to save + log.
python measurement/measure_cli.py --image PHOTO.jpg --Z 2380 \
    --log --object "hardback book" --dimension H --truth 240.0
```

### Step 3 — Validation (Part C)

```bash
# Error statistics + plots from validation/data/measurements.csv.
python validation/analyze.py

# Re-derive measured_mm from the stored pixel coordinates using the current
# geometry.py -- this is why u1..v2 are in the CSV schema.
python validation/analyze.py --recompute
```

Outputs: `validation/output/statistics.txt`, `error_vs_size.png`, `error_vs_Z.png`,
`error_distribution.png`, `measurements_with_errors.csv`.

### Verify everything

```bash
python tools/verify_pipeline.py
```

Scores all 25 checks against the synthetic ground truth — calibration accuracy, geometry,
the board self-test, pure code error, every guard, and CLI/web-app agreement. **If this
passes and your real measurements are still off, the problem is in the capture (Z origin,
resolution, focus), not in the code.**

### Regenerate the synthetic data

```bash
python tools/make_synthetic_data.py              # board + measurement scenes
python tools/simulate_measurements.py            # the 20-row CSV
python tools/simulate_measurements.py --sigma-click 0 \
    --out validation/data/measurements_bias_only.csv   # isolate the Z bias
```

---

## Capture protocol — do this before shooting

Calibration quality is decided here, not in the code.

**The board.** Print a 9×6 *inner corner* checkerboard and **mount it on something
rigid** — foam board or glass. A sheet of paper that bows by 2 mm quietly corrupts the
focal length. Measure the square size with calipers and put that number in
`--square-size`.

**The camera.**

- **Lock focus and exposure** (tap-and-hold for AE/AF Lock on iOS). If autofocus hunts
  between shots, the focal length changes shot to shot and one `K` is being fitted to
  several different cameras.
- **Never touch zoom.** Digital zoom is a crop and rescale; it changes `K`.
- **One resolution — the same one used in Steps 2 and 3.** Calibrating at 4032×3024 and
  measuring on a 1920×1080 frame makes `K` wrong by the scale factor. This is the most
  likely silent failure in the whole assignment, which is why `check_resolution()`
  refuses outright rather than warning.

**The shot list** (15–20 images): tilt the board ±30–45° in both axes; fill different
parts of the frame, **corners especially**, since that is where distortion lives; vary the
distance. Avoid a fronto-parallel board in every shot — that is a degenerate
configuration, the same coplanarity issue that makes a single view insufficient.

**Getting `Z` right** (Step 3). `Z` is the perpendicular distance from the camera's
**optical centre** to the object plane. Three ways it goes wrong:

- *Wrong origin.* A tape measure starts at the phone's back glass; the optical centre sits
  inside the lens stack, 5–10 mm away. At 2 m that is a ~0.5% systematic bias on every
  one of the 20 samples. This repo demonstrates the effect deliberately — see
  [Results](#results-on-the-synthetic-data).
- *Wrong direction.* If the object sits off to the side, the straight-line distance is
  `Z/cos θ`, not `Z`. Fix: centre the object in frame for every measurement.
- *Wrong surface.* Measure to the face being measured, not the object's front edge or base.

**Logging** (Step 3). 20 measurements means 20 measurements: vary the object *and* vary
which dimension you measure — width of one, height of another, diagonal of a third.
Measure ground truth with a ruler **before** imaging, or you will unconsciously fit it to
the computed value. Record at capture time into `validation/data/measurements.csv`:

```csv
id,image_file,object,dimension,Z_mm,ground_truth_mm,measured_mm,u1,v1,u2,v2
```

Storing the pixel coordinates means any row can be recomputed with
`analyze.py --recompute` if a bug turns up — without reshooting.

---

## The math

Forward pinhole projection:

```
u = f_x · X/Z + c_x          v = f_y · Y/Z + c_y
```

Inverted at a known depth `Z` — the only primitive the measurement code needs:

```
X = (u − c_x) · Z / f_x      Y = (v − c_y) · Z / f_y      Z = Z
```

A dimension is then the Euclidean distance between two backprojected points, so width,
height and diagonal all come from **one** function rather than three code paths. For an
axis-aligned span this collapses to the form quoted in the report:

```
W = Δu · Z / f_x             H = Δv · Z / f_y
```

**Error propagation.** Since `W = Δu · Z / f_x`, the relative errors add:

```
δW/W  ≈  δ(Δu)/Δu  +  δZ/Z  +  δf_x/f_x
```

`geometry.measure_uncertainty()` evaluates all three terms and names the dominant one.
This is why larger objects measure proportionally better: a bigger `Δu` shrinks the
click-precision term.

**Distortion — Option 1, applied in exactly one place.** The two selected points are
corrected with `cv2.undistortPoints`, which returns *normalized* coordinates (already
`X/Z` and `Y/Z`), so the dimension is `Z · Δx_normalized` with **no further division by
`f_x`**. The image itself is never undistorted — the user clicks the real, distorted
photo. Undistorting both the image and the points is the classic double-correction bug;
`geometry.py` cannot do it because only one path exists.

**Why Zhang's method rather than DLT.** A DLT fit does not model radial distortion, cannot
impose constraints such as a known focal length or zero skew, and minimises an algebraic
residual rather than geometric reprojection error. Observing one planar target in many
orientations supplies the constraints needed to separate intrinsics from per-view
extrinsics, then nonlinear refinement minimises the reprojection error that
`calibrate.py` reports.

---

## Results on the synthetic data

Reproduce with `python tools/verify_pipeline.py`.

**Step 1 — calibration vs. known ground truth** (18 images, 2016×1512):

| Parameter | Recovered | True | Error |
|---|---|---|---|
| `f_x` | 1600.290 | 1600.500 | 0.013% |
| `f_y` | 1601.076 | 1601.200 | 0.008% |
| `c_x` | 1007.258 | 1007.300 | 0.004% |
| `c_y` | 755.685 | 755.800 | 0.015% |
| `k₁` | 0.08044 | 0.08000 | 0.55% |

RMS reprojection error **0.0582 px**; `c_x/width` = 0.4996, `c_y/height` = 0.4998,
`f_y/f_x` = 1.00049 — consistent with the zero-skew, square-pixel assumption.

**Step 2 — board self-test**, 93 adjacent 25 mm gaps at a known `Z` = 1000 mm:
mean 24.997 mm, MAPE **0.07%**.

**Step 3 — the two error sources, separated.** The same 20 scenes, analysed twice:

| Run | Mean signed error | Std | MAPE | 95% CI on the mean |
|---|---|---|---|---|
| No capture noise (`--perfect`) | +0.050 mm | 0.031 mm | **0.010%** | — |
| Click noise σ=1.5 px + Z bias −8 mm | −1.170 mm | 4.357 mm | 0.633% | [−3.08, +0.74] — **contains 0** |
| Z bias −8 mm only (`--sigma-click 0`) | −1.504 mm | 1.157 mm | 0.288% | [−2.01, −1.00] — **excludes 0** |

Reading this is the point of the exercise:

- The `--perfect` row isolates **pure code error**: 0.010%, i.e. the geometry is exact and
  the residual is just the tiny remaining calibration error.
- The bias-only row shows the −8 mm `Z` origin offset exactly as predicted: a −0.288% mean
  signed error (≈ 8 mm / 2.8 m average distance), a 95% CI that excludes zero, and
  `corr(size, error)` = **−0.897** — the signature of a *scale* error rather than noise.
- The realistic row is the honest lesson: with σ = 1.5 px of click noise, the click term
  alone is ~1% on a 155 mm object, so it **swamps** the 0.3% bias and `analyze.py`
  correctly reports no statistically significant bias at n = 20. A nonzero mean signed
  error is not automatically detectable — you need either better click precision or a
  larger sample to separate bias from spread.

This is the error budget of section 12 made measurable instead of asserted.

---

## Repository layout

```
module2/
  calibration/
    capture_check.py            pre-flight photos before fitting
    calibrate.py                Step 1: Zhang calibration -> camera_params.yaml
    data/images/                board photos (synthetic_* are committed samples)
    output/                     camera_params.yaml, error CSV, debug/ overlays
  measurement/
    geometry.py                 PURE geometry: load_params, backproject, measure
    measure_cli.py              click-to-measure CLI (thin UI wrapper)
    selftest_checkerboard.py    end-to-end pipeline validation on the board
  validation/
    analyze.py                  Step 3: error statistics + plots
    data/measurements.csv       the 20 logged measurements
    output/                     statistics.txt, plots, annotated images
  theory/
    two_camera_derivation.md    Part D, typed, for the PDF
  app/
    app.py                      Flask app; all four parts from one nav bar
    templates/  static/         pages and stylesheet
  tools/
    make_synthetic_data.py      renders the synthetic dataset
    simulate_measurements.py    stands in for the 20 manual clicks
    verify_pipeline.py          scores the whole pipeline (25 checks)
  requirements.txt
  README.md                     this file
  NEXT_STEPS.md                 run sheet for capturing real data
```

**Architectural rule:** `measurement/geometry.py` has no UI dependencies — no
`cv2.imshow`, no Flask, no argparse. `measure_cli.py` and `app/app.py` both import it, so
the CLI and the web app cannot disagree; `verify_pipeline.py` asserts numerically that
they don't. The web app also imports `calibration/calibrate.py` and
`validation/analyze.py` directly rather than reimplementing them, so every number on a
web page is produced by the same code the CLI runs.

**The one real web gotcha, handled.** Canvas clicks arrive in *display* coordinates. A
2016-px-wide photo shown in an 800-px canvas needs every click multiplied by 2.52 before
it reaches the geometry, or every measurement is wrong by that factor. The browser posts
the natural dimensions alongside each click and `app.scale_click()` converts server-side,
against the server's own read of the image rather than the client's claim about it.
`verify_pipeline.py` measures the same object through both paths and asserts they match
(358.04 mm via an 800-px canvas vs. 358.00 mm ground truth).

---

## Assignment requirement → where it lives

| Requirement | Where |
|---|---|
| Step 1: calibrate a smartphone camera | `calibration/calibrate.py`, `/calibration` |
| Step 2: real-world 2D measurement from perspective projection | `measurement/geometry.py`, `/measurement` |
| Step 3: object > 2 m, 20 measurements, error statistics | `validation/`, `/validation` |
| Part D: two-camera derivation, typed | `theory/two_camera_derivation.md`, `/theory` |
| Web app reaching all assignments | `app/app.py` — one nav bar, four pages |
| ReadMe documentation at the top of each script | `HOW TO RUN` block in every module docstring |
| Error estimate statistics | `validation/output/statistics.txt` |

---

## Submission checklist

Code and app — done in this repo:

- [x] Web app reaches all four parts from one page
- [x] `HOW TO RUN` documentation at the top of every script
- [x] Geometry has no UI dependencies and is importable standalone
- [x] Resolution guard implemented and tested against a mismatched image
- [x] Distortion approach chosen, documented, applied in exactly one place
- [x] Annotated output images saved automatically
- [x] Error statistics computed reproducibly from the CSV
- [x] Theory derivation typed (not photographed)

Still requires you and a camera:

- [ ] 15+ **real** board images accepted, RMS under ~0.5 px
- [ ] Board self-test passes on a **real** photo at a tape-measured `Z`
- [ ] 20 **real** measurements beyond 2 m, varied objects and dimensions, ruler ground
      truth recorded before imaging
- [ ] Record the phone model, capture app, and resolution in `--notes` (without them `K`
      is unreproducible)
- [ ] Verify the web app and the CLI give the same number on one real object **before**
      recording
- [ ] Screen recording of the working system
- [ ] PDF with the derivation embedded and the GitHub link included
- [ ] PDF and video uploaded to Google Classroom
