# Completing Module 2 with real data

Everything in this repo is written and verified. What remains is capture: photographing a
checkerboard, photographing 20 objects, and recording the demo. This file is the
run sheet for that.

**The code does not change.** You replace three sets of files and re-run the same
commands.

---

## Contents

- [What you need](#what-you-need)
- [Time budget](#time-budget)
- [Phase 0 — Clear out the synthetic data](#phase-0--clear-out-the-synthetic-data)
- [Phase 1 — Calibration capture and fit](#phase-1--calibration-capture-and-fit)
- [Phase 2 — The self-test gate](#phase-2--the-self-test-gate-do-not-skip)
- [Phase 3 — The 20 measurements](#phase-3--the-20-measurements)
- [Phase 4 — Analysis](#phase-4--analysis)
- [Phase 5 — Cross-check the web app](#phase-5--cross-check-the-web-app-against-the-cli)
- [Phase 6 — Screen recording](#phase-6--screen-recording)
- [Phase 7 — The PDF](#phase-7--the-pdf)
- [Troubleshooting](#troubleshooting)
- [Final checklist](#final-checklist)

---

## What you need

**Printed target.** A 9×6 *inner corner* checkerboard. Nine and six count the interior
corner intersections, not the squares — a 9×6 inner-corner board has 10×7 squares. Print
it as large as your printer allows; A4 with 25 mm squares is fine.

> Generate one at <https://calib.io/pages/camera-calibration-pattern-generator> — set
> 10 columns × 7 rows of squares, which gives the 9×6 inner corners this code expects.

**A rigid backing.** Foam board, clipboard, glass, or stiff cardboard, plus glue stick or
spray adhesive. **This is not optional.** A page that bows by 2 mm corrupts the focal
length estimate, and nothing downstream will tell you that is what went wrong.

**Calipers or a steel ruler** to measure the printed square size. Do not trust the
nominal value — printers scale by a percent or two. Measure across 10 squares and divide
by 10; that divides your reading error by ten as well.

**A tape measure** with a millimetre scale, for `Z`. Something that locks is much easier
to use alone.

**A ruler or tape for ground truth** on the 20 objects.

**Your phone**, and something to steady it. A tripod is ideal. Bracing against a doorframe
or stacking books works.

**20 objects with at least one flat face.** Books, boxes, laptops, monitors, picture
frames, doors, whiteboards, cushions, a pizza box. You need flat faces because the whole
method assumes the measured face is fronto-parallel and at a single depth.

---

## Time budget

| Phase | Time | Notes |
|---|---|---|
| 0 — clear synthetic data | 5 min | |
| 1 — calibration capture + fit | 45 min | Most of it is shooting 20 photos carefully |
| 2 — self-test gate | 15 min | Catches disasters before they cost you 20 measurements |
| 3 — 20 measurements | 2–3 hrs | The bulk. Ground truth first, then image, then click |
| 4 — analysis | 10 min | One command |
| 5 — web/CLI cross-check | 10 min | Do it before recording |
| 6 — screen recording | 30 min | Including one or two retakes |
| 7 — PDF | 1–2 hrs | Writing, not computing |

Phases 1–2 are one sitting. Do not start Phase 3 until Phase 2 passes.

---

## Phase 0 — Clear out the synthetic data

```bash
cd module2
source .venv/bin/activate          # if you made one; otherwise skip

# Confirm the code passes before you change anything. 25/25 expected.
python tools/verify_pipeline.py
```

Run that first. If it passes now and something breaks later, the cause is your data, not
the code — which is a much faster thing to debug.

Then clear the rendered files:

```bash
rm -f calibration/data/images/synthetic_*
rm -f calibration/data/synthetic_ground_truth.yaml
rm -f measurement/data/synthetic_*
rm -f validation/data/images/synthetic_*
rm -f validation/data/synthetic_truth.csv
rm -f validation/data/measurements.csv validation/data/measurements_bias_only.csv
rm -rf validation/output/* calibration/output/*
```

Keep `README_SYNTHETIC.txt` in the two image folders or delete it — either is fine, the
scripts ignore non-image files.

> Want the synthetic set back for comparison? `python tools/make_synthetic_data.py`
> rebuilds it byte-identically; the generator is seeded.

---

## Phase 1 — Calibration capture and fit

### 1a. Lock the camera down

Decide these once and never change them again — not between calibration and measurement,
not between measurements:

- **One resolution.** Set it in the camera app and leave it. Note the number; you will
  put it in `--notes`.
- **Lock focus and exposure.** On iOS, tap and hold until *AE/AF LOCK* appears. On
  Android, use Pro/Manual mode and set focus to a fixed distance. If autofocus hunts
  between shots, the focal length changes shot to shot and you are fitting one `K` to
  several different cameras.
- **Zoom at 1.0×.** Never touch it. Digital zoom is a crop and rescale — it changes `K`.
  On a multi-lens phone, also make sure it does not silently switch lenses; stay at the
  main lens.
- **Turn off** any "macro mode", HDR that composites multiple exposures, or beauty/scene
  correction that warps geometry.

### 1b. Measure the square size

Measure across 10 squares, divide by 10, record in millimetres. If 10 squares span
249 mm, your square size is `24.9`, not `25.0`. This number scales every measurement you
will make, so getting it right here is worth two minutes.

### 1c. Shoot 18–20 photos

The board must be **fully visible with all 54 inner corners in frame** in every shot, or
the detector rejects it.

Vary three things:

1. **Tilt** — ±30–45° in both axes. Tilt is what lets the fit separate focal length from
   distance; a board parallel to the sensor in every shot is a degenerate configuration
   and the fit will be unstable.
2. **Position in frame** — centre, all four corners, all four edges. Frame corners matter
   most, because that is where distortion is largest and where the coefficients get
   pinned down.
3. **Distance** — some filling most of the frame, some smaller.

A shot list that works: 9 positions (3×3 grid) × 2 tilts each, plus 2 extra. Keep the
board still and move yourself, or keep yourself still and move the board — either way,
avoid motion blur.

Copy them in:

```bash
cp /path/to/photos/*.jpg calibration/data/images/
```

### 1d. Pre-flight before fitting

```bash
python calibration/capture_check.py
```

This is a 10-second check that saves a reshoot. You want:

- **detected ≥ 15.** Below that, shoot more.
- **frame coverage ≥ 60%** of the 3×3 grid, ideally 100%.
- **blur above ~100.** Phone photos of a sharp printed board score in the hundreds or
  thousands. Anything under 100 is soft — check focus lock.
- **a single resolution.** If it reports mixed resolutions, something changed mid-shoot.

Fix problems now, not after fitting.

### 1e. Fit

```bash
python calibration/calibrate.py \
    --square-size 24.9 \
    --pattern 9x6 \
    --notes "iPhone 14 Pro, native camera, 4032x3024, AE/AF locked, 1.0x"
```

Replace the square size and notes with yours. **Record the phone model, app, and
resolution** — without them `K` is unreproducible, and the report needs it.

Read the summary against these targets:

| Reported value | Target | If it is off |
|---|---|---|
| images used | ≥ 15 | Shoot more; check which were rejected |
| RMS reprojection error | **< 0.5 px** | See below |
| `f_y / f_x` | within 1% of 1.0 | Usually fine; a big deviation suggests a non-square crop |
| `c_x / width`, `c_y / height` | 0.45–0.55 | If near 0.4, suspect a crop or systematic detection failure |

**On RMS.** Under 0.5 px is the goal. Between 0.5 and 1.0 is usable but worth improving.
Above 1.0 px something is wrong upstream: blurry shots, a bowed board, or the wrong
`--pattern`. Look at the per-image error table the script prints — if one photo is far
worse than the rest, delete it and refit:

```bash
rm calibration/data/images/IMG_1021.jpg
python calibration/calibrate.py --square-size 24.9 --notes "..."
```

Do not delete photos just to drive the number down. Removing one clear outlier is
legitimate; pruning to the best eight is fitting to noise, and it will show up as a worse
result in Phase 3.

Check the corner overlays in `calibration/output/debug/` — the drawn grid should sit
exactly on the printed corners. These images go in the PDF.

---

## Phase 2 — The self-test gate. Do not skip.

You now have a `K`. Before spending three hours on 20 measurements, confirm the whole
pipeline produces correct millimetres — using the board, whose true dimensions you
already know.

**Shoot one photo:** board flat against a wall, camera square to it (not tilted), board
centred in frame, at a tape-measured distance of about 1000 mm. Same camera settings as
Phase 1.

Measure `Z` from the **wall surface to the phone's lens**, and add about 5 mm to account
for the optical centre sitting inside the lens stack rather than at the glass. That
offset is the systematic bias this project demonstrates; see
[Getting Z right](#getting-z-right-the-single-biggest-error-source) below.

```bash
cp /path/to/board_at_1m.jpg measurement/data/
python measurement/selftest_checkerboard.py \
    --image measurement/data/board_at_1m.jpg --Z 1000
```

This measures every adjacent corner gap on the board — about 93 independent measurements
from one photo — and compares each to your true square size.

**How to read the result:**

| Outcome | Meaning |
|---|---|
| Within 1–2% | Working. Proceed to Phase 3. |
| 3–5% | Passes, but check your `Z` measurement and whether the board was really square to the camera. |
| ~25% high or low | Almost certainly a resolution mismatch. |
| A clean ratio like 1.26× or 2× off | Resolution mismatch, or `--square-size` does not match the printed board. |
| Fails only in one axis | The board was tilted, not fronto-parallel. Reshoot. |

A 25 mm square reading 25.3 mm means the pipeline is good. A 25 mm square reading 31 mm
means stop and find the bug — you have just saved yourself 20 wasted measurements.

---

## Phase 3 — The 20 measurements

### Getting Z right: the single biggest error source

`Z` is the perpendicular distance from the camera's **optical centre** to the object
plane. Three ways it goes wrong, in order of how often:

1. **Wrong origin.** A tape measure starts at the phone's back glass or at the screen
   edge. The optical centre sits inside the lens stack, roughly 5–10 mm further in. At
   2 m that is a ~0.4% bias on *every single measurement* — it will not average out, and
   it will show up in Phase 4 as a nonzero mean signed error. Pick a convention, write it
   down, and apply it to all 20. Adding a constant 7 mm to a tape reading taken at the
   back glass is a reasonable convention.
2. **Wrong direction.** If the object sits off to the side of the frame, the straight-line
   camera-to-object distance is `Z / cos θ`, not `Z`. **Fix: centre the object in the
   frame for every measurement.** This costs nothing and removes the error entirely.
3. **Wrong surface.** Measure to the *face you are measuring*, not the object's front edge
   or its base. For a book standing upright, that is the cover, not the spine.

### The protocol, per object

Do these in order. The order matters.

1. **Measure ground truth first, with a ruler.** Write it down. Measuring after you have
   seen the computed value invites unconscious fitting.
2. **Place the object** beyond 2 m, flat face toward the camera, centred in frame.
3. **Measure `Z`** with the tape, applying your origin convention.
4. **Shoot**, with the same locked camera settings.
5. **Note the filename** against the row in your paper log.

Vary the objects **and** vary which dimension you measure — width of one, height of
another, diagonal of a third. Twenty clicks on the same book is not 20 measurements. Vary
`Z` too, from just over 2 m out to 4 m or more; that spread is what makes the
error-vs-distance plot in Phase 4 informative.

Aim for a mix like 8 widths, 7 heights, 5 diagonals across 12–15 distinct objects.

### Clicking and logging

Copy the photos in, then measure each one with `--log`, which appends the row to the CSV
directly — no pixel coordinate is ever retyped:

```bash
cp /path/to/measurement_photos/*.jpg validation/data/images/

python measurement/measure_cli.py \
    --image validation/data/images/IMG_1042.jpg \
    --Z 2387 \
    --log --object "hardback book" --dimension H --truth 240.0
```

A window opens. Click the two endpoints of the dimension. The measurement prints
immediately. Press **`s`** to save the annotated image *and* append the row; **`r`** to
reset if you misclicked; **`q`** to move on.

Keys: `r` = reset, `s` = save + log, `q`/`Esc` = quit.

Row ids auto-increment, so just run the command once per photo and the CSV builds itself.

**Click carefully.** Click precision is usually the dominant error term for smaller
objects — on a 155 mm object at 2.5 m, two pixels of click error is about 1%. Zoom your
OS window manager in if it helps. Click the same feature at both ends: outer edge to
outer edge, not outer edge to inner edge.

If you would rather click in the browser, the web app's Measurement page does the same
thing, but it does not append to the CSV — use the CLI for the 20 logged rows and the web
app for the demo.

### Check the CSV as you go

```bash
column -s, -t validation/data/measurements.csv
```

You are looking for 20 rows, sane `Z` values, and `measured_mm` in the same ballpark as
`ground_truth_mm`. If one row is wildly off, you can delete that line and redo just that
photo.

---

## Phase 4 — Analysis

```bash
python validation/analyze.py
```

This writes `validation/output/statistics.txt` plus three plots, and prints everything the
report needs: mean signed error, standard deviation, MAE, RMSE, percent errors, MAPE,
min/max with the responsible measurement named, a 95% confidence interval on the mean, and
two trend correlations.

### How to interpret what you get

**Mean signed error** is your systematic bias. Look at whether the 95% CI excludes zero:

- **CI excludes zero** → the bias is real, not sampling noise, and the report must
  attribute it rather than call it noise. The leading suspect is the `Z` origin
  convention; the other is a residual focal-length error from Phase 1.
- **CI contains zero** → no statistically detectable bias at n=20. That usually means
  click precision is dominating. This is a legitimate finding, not a failure — say so and
  back it with the error-propagation argument.

**`corr(ground-truth size, signed error)`**, printed under Trend Diagnostics. A strong
value means error grows in proportion to size, which is the signature of a **scale**
error — `f_x` or `Z` — rather than click noise.

**`corr(Z, percent error)`.** A trend here implicates the `Z` measurement itself: a fixed
origin offset `δZ` produces a percent error `δZ/Z` that shrinks with distance, so it
shows up as a correlation against `Z`.

**Typical honest numbers** for careful work with a phone: MAPE around 1–3%, with the
larger objects doing better than the small ones. If you are seeing under 0.5%, double-check
that your ground truth was really measured independently. If you are seeing over 10%,
work through the [Troubleshooting](#troubleshooting) table.

Re-derive results from the stored pixels at any time:

```bash
python validation/analyze.py --recompute
```

---

## Phase 5 — Cross-check the web app against the CLI

Do this **before** recording. A browser scales a large photo down to fit the canvas, so a
click arrives in display coordinates — if that is not converted back to full resolution,
every web measurement is wrong by the scale factor. The code handles this and
`verify_pipeline.py` asserts it, but confirm it on *your* data:

```bash
# 1. CLI, on a known photo with known pixel coordinates
python measurement/measure_cli.py --image validation/data/images/IMG_1042.jpg \
    --Z 2387 --points 812.5,431.0 1904.0,438.5

# 2. Same photo, same Z, same two points clicked in the browser
python app/app.py
# open http://127.0.0.1:5000/measurement
```

The two numbers should agree to within your click precision — a millimetre or two. If the
web number is off by a large ratio (roughly `image_width / canvas_width`), stop and say
so; that is the one bug that silently invalidates the demo.

---

## Phase 6 — Screen recording

**Record with:** macOS `Cmd+Shift+5`, or QuickTime → New Screen Recording. Windows: Xbox
Game Bar (`Win+G`) or OBS. Record audio narration if you can; it makes a large difference
to how a demo reads.

**Before you hit record:** close other windows, make the browser window large, zoom the
page to about 110% so text is readable in the video, and have your files and `Z` values
to hand so you are not hunting mid-take.

**A running order that covers everything in 4–6 minutes:**

1. **Overview page** (20 s). Show that the nav bar reaches all four parts, and point out
   the loaded camera parameters.
2. **Calibration page** (90 s). Upload your board images live. Let the fit run. Walk
   through `K`, the distortion coefficients, the RMS error, the exit-criteria table, the
   per-image error table, and the corner overlays. Download the YAML to show it works.
3. **Measurement page** (90 s). Upload one measurement photo, type its `Z`, click two
   points. Read out width, height, diagonal. Point at the error budget and name the
   dominant term. Show the annotated output.
4. **Validation page** (90 s). Walk the 20-row table, then the statistics: mean signed
   error, standard deviation, MAE, RMSE, MAPE. Interpret the confidence interval out
   loud — bias or no bias, and why. Show the three plots.
5. **Theory page** (60 s). Scroll the derivation and read out the boxed result
   `x₂ ≅ H∞x₁ + e₂/Z`, explaining in words that the second view sees the point displaced
   along the epipole by an amount inversely proportional to depth — which is why one view
   cannot give you depth and you needed the tape measure.
6. **Terminal** (30 s, optional but strong). Run `python tools/verify_pipeline.py` and let
   the 25/25 scroll past.

Save the file somewhere you will find it. Google Classroom has upload size limits — if the
file is large, upload to Drive and submit the link.

---

## Phase 7 — The PDF

Required contents, per the assignment:

1. **The GitHub repo link.** Push first, then paste the URL.
2. **Step 1 — calibration.** Your `K`, the distortion coefficients, the RMS error, the
   number of images, and the per-image error table. Two or three corner overlays. State
   the phone model, capture app, and resolution. State that skew is assumed zero and
   pixels square, and cite your measured `f_y/f_x` as support. Note that
   `cv2.calibrateCamera` implements Zhang's multi-plane method, and say why that is
   preferred over a DLT fit: DLT models no radial distortion, cannot impose constraints
   such as a known focal length, and minimises an algebraic rather than a geometric error.
3. **Step 2 — the method.** The forward projection equations, the inverted form, and the
   `W = Δu · Z / f_x` collapse. State that you undistort the two selected points only, via
   `undistortPoints`, and that the image itself is never undistorted. Include one or two
   annotated measurement images.
4. **Step 3 — results.** The 20-row table, the full statistics block, and the plots.
   Then the discussion that actually earns the marks:
   - Attribute your mean signed error. Do not call it noise if the CI excludes zero.
   - Tie the observed spread to the error propagation `δW/W ≈ δ(Δu)/Δu + δZ/Z + δf_x/f_x`,
     and say which term dominated and why.
   - State the limiting assumption plainly: the measured face is fronto-parallel and at a
     single depth `Z`. Any tilt or object thickness violates it. That is your limitations
     paragraph.
5. **Part D — the derivation.** Paste in `theory/two_camera_derivation.md`, or export the
   `/theory` page to PDF from the browser. **Typed or scanned only** — a photo of a paper
   worksheet is explicitly not accepted.
6. **A note on the web app**, with the launch command.

Getting the theory page into the PDF is easiest from the browser: open
`http://127.0.0.1:5000/theory`, then Print → Save as PDF.

---

## Troubleshooting

### Calibration

| Symptom | Cause | Fix |
|---|---|---|
| "corners not found" on most images | Wrong `--pattern`; you counted squares, not inner corners | A board with 10×7 squares is `--pattern 9x6` |
| Corners found on some, not others | Board cut off at the frame edge, blur, or glare | All 54 corners must be in frame; diffuse light, no flash |
| RMS above 1.0 px | Bowed board, soft focus, or one bad photo | Mount rigidly; check the per-image table and drop the worst outlier |
| `c_x/width` near 0.4 | The camera app cropped, or detection is failing systematically | Check for aspect-ratio or HDR crop; refit |
| `f_y/f_x` far from 1.0 | Non-square pixel crop, anamorphic processing | Turn off scene correction; use a plain camera mode |
| "resolution NxM != PxQ" | Mixed resolutions in the folder | One resolution only; remove the odd ones |

### Measurement

| Symptom | Cause | Fix |
|---|---|---|
| "RESOLUTION MISMATCH … refused" | Measurement photo differs from calibration resolution | Reshoot at the calibration resolution. `--allow-scaling` exists but is approximate and assumes a pure resize with no crop |
| Everything reads ~25% too large or small | Resolution mismatch, or wrong `--square-size` in Phase 1 | Re-check both; rerun the Phase 2 self-test |
| Everything off by a constant percentage | `Z` origin convention, or focal length | Phase 2 self-test isolates which |
| Errors scale with object size | Scale error: `f_x` or `Z` | Check `corr(size, error)` in the analysis |
| Errors random, worse on small objects | Click precision — expected behaviour | Click more carefully; prefer larger objects; report it as the dominant term |
| One row wildly wrong, others fine | Misclick, or `Z` transposed | Delete the row and redo that photo |

### Web app

| Symptom | Cause | Fix |
|---|---|---|
| "No camera parameters loaded" | Phase 1 has not been run | Run `calibrate.py`, or fit on the Calibration page |
| Web number differs hugely from CLI | Display-coordinate scaling | Should not happen — the code handles it. Rerun `verify_pipeline.py` and report it |
| Upload rejected | Not an image, or over 200 MB | Check the extension; the cap is in `app/app.py` |
| Port already in use | Something else on 5000 | `python app/app.py --port 8000` |

---

## Final checklist

Capture and data:

- [ ] Board printed at 9×6 inner corners and mounted rigidly
- [ ] Square size measured with calipers, not assumed
- [ ] Camera locked: one resolution, AE/AF locked, zoom at 1.0×, unchanged throughout
- [ ] 15+ board images accepted, RMS under ~0.5 px
- [ ] `c_x/width` and `c_y/height` both near 0.5
- [ ] Phase 2 self-test passes within a few percent
- [ ] 20 measurements, all beyond 2 m, varied objects **and** varied dimensions
- [ ] Ground truth measured with a ruler **before** imaging, every time
- [ ] `Z` origin convention chosen, written down, applied consistently to all 20
- [ ] Object centred in frame for every measurement
- [ ] All 20 rows in `validation/data/measurements.csv`, pixel coordinates included
- [ ] Every raw image kept, named to match its CSV row

Verification:

- [ ] `python tools/verify_pipeline.py` → 25/25
- [ ] `python validation/analyze.py` runs clean and reports 20 rows
- [ ] Web app and CLI agree on one real object

Submission:

- [ ] GitHub repo pushed and accessible; link in the PDF
- [ ] Web app reaches all four parts from one page
- [ ] Screen recording captured
- [ ] Theory derivation typed or scanned, embedded in the PDF
- [ ] Error statistics for all 20 measurements reported **and interpreted**
- [ ] Mean signed error attributed to a cause, not dismissed as noise
- [ ] Limitations paragraph states the fronto-parallel, single-depth assumption
- [ ] PDF and video uploaded to Google Classroom

---

## Quick command reference

```bash
cd module2

# Phase 0
python tools/verify_pipeline.py

# Phase 1
python calibration/capture_check.py
python calibration/calibrate.py --square-size 24.9 --notes "PHONE, APP, WxH, locked"

# Phase 2
python measurement/selftest_checkerboard.py --image measurement/data/board.jpg --Z 1000

# Phase 3, once per object
python measurement/measure_cli.py --image validation/data/images/IMG_xxxx.jpg \
    --Z 2387 --log --object "NAME" --dimension H --truth 240.0

# Phase 4
python validation/analyze.py

# Phases 5-6
python app/app.py
```
