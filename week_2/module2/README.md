# Module 2 — Camera Calibration and Real-World Measurement

**CSc 8830: Computer Vision** · Python + OpenCV + Flask

Camera calibration by Zhang's multi-plane method, real-world 2D measurement by inverting
the perspective projection at a known object distance, a 20-measurement validation
experiment with error statistics, the two-camera derivation, and a web application that
reaches all four parts from one page.

---

## The data here is real

The experiment has been run. `calibration/output/camera_params.yaml` was fitted from 24
checkerboard photos of an actual phone camera at 3024×4032, and
`validation/data/measurements.csv` holds 20 measurements of real objects against ruler
ground truth. The rendered `synthetic_*` stand-in set that this repo used to ship — and
the tools that generated and scored it — were removed once real captures replaced them;
they are in git history if ever needed.

The write-up of Steps 1–3, including the error attribution, is
**[`report/report.tex`](report/report.tex)**. Numbers quoted below are regenerated from
the committed CSV by `validation/analyze.py`.

---

## Install

```bash
cd week_2/module2
python -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt
```

Verified on Python 3.9.23 with opencv-python 5.0.0, numpy 2.0.2, Flask 3.1.3,
matplotlib 3.9.4.

---

## Launch the web application

**The web app now lives at the repository root, not in this folder**, so one server
reaches every weekly assignment. Module 2's pages are a blueprint inside it
(`app/modules/module2.py`) and still import this folder's code, so nothing about the
numbers changed — only where the server starts:

```bash
cd ../..            # the repository root
python app/app.py
# open http://127.0.0.1:5000        -> every assignment
#      http://127.0.0.1:5000/module-2/   -> this one
```

Options: `--port 8000`, `--host 0.0.0.0`, `--debug`.

The header switcher moves between assignments; the nav bar below it reaches all four
parts of this one, so a screen recording can walk the whole assignment without retyping
URLs:

| Page | Part | What it does |
|---|---|---|
| `/` | — | Home: every assignment in the repository |
| `/module-2/` | A · Step 1 | Calibration: upload board images → fit `K` and distortion → per-image error table → corner overlays → download `camera_params.yaml` |
| `/module-2/measure` | B · Step 2 | Upload an image, enter `Z`, click two points → width/height/diagonal + error budget + annotated image |
| `/module-2/records` | C · Step 3 | Measurement table, error statistics, diagnostic plots, report figures `.zip` |

Part D is written rather than interactive: [`theory/two_camera_derivation.md`](theory/two_camera_derivation.md).

---

## Running the pipeline

Every script carries its own `HOW TO RUN` block in its module docstring. Run all commands
**from `week_2/module2/`** — except the web app, which starts from the repository root.

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
# Photograph the board fronto-parallel at a tape-measured distance, then:
python measurement/selftest_checkerboard.py --image board_at_1m.jpg --Z 1000

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

### Verify a change

The synthetic-scoring suite is gone: it compared the calibration against rendered ground
truth, which stopped meaning anything once the real captures replaced it. Two checks
remain, and both matter after touching anything numeric.

```bash
python measurement/geometry.py             # pure geometry + every guard, no data needed
python validation/analyze.py --recompute   # re-derive every logged row from its pixels
```

`--recompute` must reproduce the logged `measured_mm` to rounding (~0.005 mm). A larger
drift means the geometry or the calibration has changed under the CSV, and the numbers in
`report/` and on the Records page no longer agree with it.

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
  sample. Worth knowing, but note that it is *not* what happened here: a fixed offset
  produces a percent error that shrinks with distance, and the measured data shows a
  constant proportional error instead — see [Results](#results).
- *Wrong magnitude.* Setting `Z` once from a nominal distance and reusing it for every
  shot produces a proportional error on all of them. This is the leading explanation for
  the 10.3% scale deficit in [Results](#results); measure `Z` per shot and record what
  the tape actually read.
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

## Results

Regenerate with `python validation/analyze.py`.

**Step 1 — calibration** (24 images, 3024×4032):

| Parameter | Value |
|---|---|
| `f_x` | 3336.69 px |
| `f_y` | 3335.58 px |
| `c_x` | 1496.68 px |
| `c_y` | 2020.27 px |
| `f_y/f_x` | 0.99967 |
| Overall RMS reprojection error | 1.863 px |
| Median per-image RMS | 0.581 px |

`f_y/f_x` within 0.03% of unity and a principal point within 1% of the image centre are
both consistent with the zero-skew, square-pixel assumption. The overall RMS is not
representative: 20 of the 24 images sit below 1 px (median 0.533 px) and four land at
1.22, 2.30, 3.72 and 7.48 px. Those four dominate the total and refitting without them is
the cheapest improvement available here.

**Step 3 — validation**, 20 measurements, 77–413 mm, `Z` = 1–2 m:

| Statistic | Value |
|---|---|
| Mean signed error | −19.64 mm |
| Std deviation | 19.91 mm |
| MAE | 23.52 mm |
| RMSE | 27.61 mm |
| Mean signed percent error | −7.77% |
| MAPE | 10.76% |
| 95% CI on the mean signed error | [−28.36, −10.91] mm |
| `corr(size, signed error)` | −0.721 |
| `corr(Z, percent error)` | +0.071 |

The CI excludes zero, so the bias is real rather than sampling noise. It is
**multiplicative, not additive**: fitting one through-origin scale factor to the
consistent rows gives `measured = 0.8974 × truth`, and dividing that single constant out
drops the MAPE from 9.75% to 4.02%.

A fixed `Z` origin offset — the usual suspect — is ruled out by the data. A fixed offset
producing −9.75% at `Z` = 2000 mm would produce −16.32% at `Z` = 1000 mm; the `Z` = 1000
row reads −10.76%, essentially the same. That leaves `f_x` being 10.3% high or `Z` being
10.3% low, and the evidence favours `Z`: `f_x/W` = 1.103 is the textbook value for a phone
main camera, and `Z` is recorded as exactly 2000.0 in 19 of 20 rows, i.e. a nominal setup
distance rather than a per-shot measurement.

Row 4 is a ground-truth entry error, not a measurement error: the stored pixels genuinely
mean 168.8 mm against a recorded truth of 130 mm. It is kept in every statistic above
rather than dropped, and it alone moves the mean signed percent error from −9.75% to
−7.77%.

The full attribution, with the figures, is in [`report/report.tex`](report/report.tex).

---

## Repository layout

```
CSC8830_Assignments/
  app/                          the web application, one server for every assignment
    app.py                      Flask instance, home page, shared navigation
    assignments.py              the registry: which assignments exist and where
    modules/module2.py          Module 2's pages (this folder), as a blueprint
    templates/  static/         base chrome + home page; templates/module2/ per page
  week_2/module2/
  calibration/
    capture_check.py            pre-flight photos before fitting
    calibrate.py                Step 1: Zhang calibration -> camera_params.yaml
    data/images/                board photos (populated by an upload)
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
  report/
    report.tex                  the Steps 1-3 write-up
    figures/                    the figures it includes
  requirements.txt
  README.md                     this file
  NEXT_STEPS.md                 run sheet used to capture the real data
```

**Architectural rule:** `measurement/geometry.py` has no UI dependencies — no
`cv2.imshow`, no Flask, no argparse. `measure_cli.py` and `app/modules/module2.py` both
import it, so
the CLI and the web app cannot disagree. The web app also imports `calibration/calibrate.py` and
`validation/analyze.py` directly rather than reimplementing them, so every number on a
web page is produced by the same code the CLI runs.

**The one real web gotcha, handled.** Canvas clicks arrive in *display* coordinates. A
3024-px-wide photo shown in an 800-px canvas needs every click multiplied by 3.78 before
it reaches the geometry, or every measurement is wrong by that factor. The browser posts
the canvas dimensions alongside each click and `modules.module2.scale_click()` converts
server-side, against the server's own read of the image rather than the client's claim
about it.

---

## Assignment requirement → where it lives

| Requirement | Where |
|---|---|
| Step 1: calibrate a smartphone camera | `calibration/calibrate.py`, `/module-2/` |
| Step 2: real-world 2D measurement from perspective projection | `measurement/geometry.py`, `/module-2/measure` |
| Step 3: object > 2 m, 20 measurements, error statistics | `validation/`, `/module-2/records` |
| Part D: two-camera derivation, typed | `theory/two_camera_derivation.md` |
| Write-up of Steps 1-3 | `report/report.tex` |
| Web app reaching all assignments | `app/` at the repository root — one home page, one nav bar per assignment |
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
