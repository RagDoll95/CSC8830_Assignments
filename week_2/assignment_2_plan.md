# Module 2 — Implementation Plan

**Course:** CSc 8830: Computer Vision — Module 2 Assignment
**Language:** Python (OpenCV)

**Contents**

- Part A — Step 1: Camera calibration
- Part B — Step 2: Real-world 2D measurement from perspective projection

---

# Part A — Step 1: Camera Calibration

**Goal:** Recover intrinsics `K` and distortion coefficients for a smartphone camera, saved to disk for reuse in Steps 2 and 3.

---

## 1. Capture protocol

This determines whether the numbers are any good, and it is the part that is painful to redo.

### Target

- Print a checkerboard with **9×6 inner corners** (the standard OpenCV pattern).
- **Mount it on something rigid** — foam board or glass. A sheet of paper that bows by 2 mm will quietly corrupt the focal length estimate.
- Measure the square size with calipers or a good ruler. Record it in mm.

### Camera settings

- **Lock focus and exposure.** Tap-and-hold on iOS (AE/AF Lock), or use an app with manual mode. If autofocus hunts between shots, the focal length changes shot to shot and one `K` is being fit to several different cameras.
- **Never touch zoom.** Digital zoom is a crop and rescale; it changes `K`.
- **One resolution — the same resolution used in Steps 2 and 3.** Calibrating at 4032×3024 and then measuring on a 1920×1080 frame makes `K` wrong by the scale factor. Decide the resolution before shooting.

### Shot list (15–20 images)

- Tilt the board ±30–45° in both axes.
- Fill different parts of the frame, corners especially — that is where distortion lives.
- Vary the distance to the board.
- Avoid a fronto-parallel board in every shot. That is a degenerate configuration, the same underlying issue as the coplanar-point degeneracy noted in the lecture slides.

---

## 2. Repo structure

```
module2/
  calibration/
    capture_check.py      # sanity-check images before fitting
    calibrate.py          # the main routine
    data/images/          # raw calibration photos
    output/               # camera_params.yaml, debug overlays
  measurement/            # Step 2
  app/                    # Flask/Streamlit wrapper
  README.md
```

---

## 3. `calibrate.py` — what to write

### 3.1 Config block at top

`PATTERN_SIZE = (9, 6)`, `SQUARE_SIZE_MM = 25.0`, image glob path, output path. Keep these in one place so the README can explain them.

### 3.2 Build the object-point template once

A `(54, 3)` float32 array of the form `(col·s, row·s, 0)` — the board defines the world frame with Z = 0. `np.mgrid` does this in two lines. Multiply by `SQUARE_SIZE_MM` so `K` comes out on a real metric scale and Step 3 outputs are in mm.

### 3.3 Loop over images

For each image:

1. Convert to grayscale.
2. `findChessboardCorners` with `CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE`.
3. On failure, **log the filename and skip** rather than crashing — it matters which shots were rejected.
4. On success, refine with `cornerSubPix` (11×11 window, 30 iterations, eps 0.001). Skipping subpixel refinement is the single most common way people end up with a mediocre reprojection error.

Also assert every image has identical shape, and store that shape for the `calibrateCamera` call.

### 3.4 Write debug overlays

`drawChessboardCorners` on each accepted image, saved to `output/debug/`. Nearly free, and it is strong evidence for the PDF and the demo video.

### 3.5 Fit

```
cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
```

Returns RMS reprojection error, `K`, distortion coefficients, and per-image `rvecs`/`tvecs`.

- Target RMS: **under ~0.5 px**.
- Above ~1.0 px, something is wrong upstream — blurry shots, a soft board, or the wrong pattern size.

### 3.6 Compute per-image reprojection error

Run `projectPoints` with each `rvec`/`tvec`, then take the L2 norm against the detected corners. Print a sorted table. This finds the one bad photo dragging the fit down, and the table is worth including in the report.

### 3.7 Save to YAML

Use `cv2.FileStorage` (or JSON). Store:

- `K`
- distortion coefficients
- image size
- square size
- pattern size
- RMS error
- image count

Step 2 loads this file. Do not hardcode `K` in two places.

### 3.8 Print a human-readable summary

Report `f_x`, `f_y`, `c_x`, `c_y`, and the ratio `c_x / width`. That ratio should land near 0.5; if it is at 0.4, there is likely a crop or a detection problem.

---

## 4. Notes for the report

- Phone cameras usually have `f_x ≈ f_y` within a fraction of a percent. State that skew is assumed zero and pixels are square — this connects back to the intrinsic parameter list in the slides.
- Record the phone model, the capture app, and the resolution. Without those, `K` is unreproducible.
- Mention that OpenCV's `calibrateCamera` implements Zhang's multi-plane method, which is the "Alternative: multi-plane calibration" approach from the slides, and note why it is preferred over DLT: DLT does not model radial distortion, cannot impose constraints such as known focal length, and does not minimize the geometrically meaningful error.

---

## 5. Exit criteria before moving to Step 2

- [ ] 15+ images accepted by the corner detector
- [ ] RMS reprojection error under ~0.5 px
- [ ] `c_x / width` and `c_y / height` both near 0.5
- [ ] `camera_params.yaml` written and loadable
- [ ] Debug overlays saved for the report

---

# Part B — Step 2: Real-World 2D Measurement

**Goal:** Given a calibrated camera and a known object distance, compute real-world 2D dimensions from image coordinates.

Step 2 does not depend on the calibration being finished — it only needs to *load* a `camera_params.yaml`. It can be written and tested against a placeholder file, with the real one swapped in later.

---

## 6. The math being implemented

From the pinhole model in the slides:

```
u = f_x · X/Z + c_x        v = f_y · Y/Z + c_y
```

Inverting it: given a pixel and a known depth Z, the 3D point in the camera frame is

```
X = (u − c_x) · Z / f_x
Y = (v − c_y) · Z / f_y
Z = Z
```

So the cleanest primitive to build is **backprojection of a pixel to a 3D point at known depth**. Once that exists, a dimension between two selected points is just the Euclidean distance between their backprojections — width, height, and diagonal all fall out of one function instead of needing three code paths.

For an axis-aligned span this collapses to

```
W = Δu · Z / f_x        H = Δv · Z / f_y
```

which is the form to show in the report, since the assignment asks for "perspective projection equations."

---

## 7. Architecture

The web-app requirement is coming, so write the geometry as pure functions with no UI and no OpenCV windows inside them:

```
measurement/
  geometry.py       # load_params, backproject, measure
  measure_cli.py    # cv2 click interface, calls geometry
```

`app/` then imports the same `geometry.py`. Baking point-picking into the measurement logic means rewriting it for the web app.

Functions worth having:

- `load_params(path) → K, dist, calib_image_size`
- `backproject(u, v, Z, K) → (X, Y, Z)` in mm
- `measure(p1, p2, Z, K, dist) → dict` with `width`, `height`, `length`, plus the intermediate pixel span

---

## 8. Handling distortion — pick one approach and stay consistent

Two valid routes. Mixing them is a classic bug.

**Option 1 (recommended): undistort the two selected points only.**
`cv2.undistortPoints(pts, K, dist)` returns *normalized* coordinates — already `X/Z` and `Y/Z` — so the dimension is just `Z · Δx_normalized`, with no further division by `f_x`. More accurate, and it matches the fact that the user clicks on the real, distorted image.

**Option 2: undistort the whole image** with `cv2.undistort`.
If `getOptimalNewCameraMatrix` is used, the intrinsics change and the **new matrix must be used in the equations**. Undistorting with a new matrix and then measuring with the old `K` is a common error.

Whichever is chosen, state it in a comment. Never undistort the image *and* the points.

---

## 9. Guards to build in

- **Assert that the measurement image resolution matches the calibration resolution.** If it differs, either refuse to proceed or scale `f_x, f_y, c_x, c_y` by the ratio and print a loud warning. This is the most likely silent failure in the whole assignment.
- Reject `Z ≤ 0` and identical selected points.
- Save an annotated output image: the two points, the connecting line, and the computed dimension drawn on it. These go straight into the PDF.

---

## 10. Defining Z correctly

`Z` is the perpendicular distance from the camera's **optical center** to the object plane. Three ways it goes wrong:

- **Wrong origin.** A tape measure starts at the phone's back glass or screen edge. The optical center sits inside the lens stack, roughly 5–10 mm off. At 2 m that is a ~0.5% systematic bias on every one of the 20 samples.
- **Wrong direction.** If the object sits off to the side of the frame, the straight-line camera-to-object distance is not `Z` — it is `Z / cos θ`. Fix: center the object in the frame for every measurement.
- **Wrong surface.** Measure to the face of the object being measured, not its front edge or its base.

---

## 11. End-to-end self-test

Make the **checkerboard itself** the first measurement target, at a tape-measured distance. The true square size in mm is already known, so it validates the whole pipeline at once.

- 25 mm square returns ~25.3 mm → pipeline is working.
- 25 mm square returns ~31 mm → resolution mismatch or units bug, found before burning 20 experimental measurements.

---

## 12. Error propagation (for the report)

Since `W = Δu · Z / f_x`, the relative errors add:

```
δW/W ≈ δ(Δu)/Δu + δZ/Z + δf_x/f_x
```

This identifies where Step 3's error budget comes from and justifies design choices — for example, preferring larger objects, since a bigger `Δu` shrinks the click-precision term. It also supplies most of the "error estimate statistics" discussion the assignment asks for.

---

## 13. Exit criteria before moving to Step 3

- [ ] `geometry.py` has no UI dependencies and is importable on its own
- [ ] Resolution guard implemented and tested by deliberately feeding a mismatched image
- [ ] Distortion approach chosen, documented, and applied in exactly one place
- [ ] Checkerboard self-test passes to within a few percent
- [ ] Annotated output images being saved automatically

---

# Part C — Step 3: Validation Experiment

**As assigned:** image an object from a chosen distance greater than 2 m, measured accurately. Validate across **20 different object measurements**. Report error estimate statistics.

Not much to build here — it is mostly careful execution. The reminders below are the things that are expensive to fix after the fact.

---

## 14. Reminders before shooting

- **Same camera configuration as calibration.** Same resolution, same lens, focus locked, no zoom. If anything changed since Step 1, recalibrate.
- **Distance > 2 m**, measured to the optical center, with the object centered in frame (see section 10).
- **20 measurements means 20 measurements.** Vary objects and vary which dimension is measured — width of one, height of another, diagonal of a third. Twenty clicks on the same book is not 20 measurements.
- **Ground truth first.** Measure each object with a ruler or tape *before* imaging it, and record it. Measuring afterward invites unconscious fitting to the computed value.
- Keep every raw image, named to match its log row.

---

## 15. Log one CSV row per measurement

Record at capture time, not from memory:

```
id, image_file, object, dimension (W/H/D), Z_mm,
ground_truth_mm, measured_mm, u1, v1, u2, v2
```

Storing the pixel coordinates means any measurement can be recomputed later if a bug in `geometry.py` turns up — without reshooting.

---

## 16. Statistics to report

Compute from the CSV in a separate `analyze.py` so the numbers are reproducible:

- Signed error per measurement: `measured − ground_truth`
- Mean signed error → reveals **systematic bias** (the Z-origin problem shows up here)
- Standard deviation of error → random spread
- MAE and RMSE
- Percent error per measurement, plus mean absolute percent error
- Min / max error, and note which measurement produced each

Plots worth including: error vs. ground-truth size, and error vs. Z. A trend in either points to a specific cause rather than noise.

---

## 17. Discussion points for the report

- A nonzero mean signed error is **not** noise. Attribute it — most likely the Z reference origin or a residual focal-length error.
- Tie the observed spread back to the error propagation in section 12.
- State the assumption the whole method rests on: the measured face is **fronto-parallel** and at a single depth `Z`. Any tilt or thickness violates it, and that belongs in the limitations paragraph.

---

# Part D — Theory: Two-Camera Relationship

**As assigned:** derive mathematically the relationship between the image coordinates of a point P(X, Y, Z) in camera 1 and camera 2, where camera 1 is static and camera 2 sits at some distance and at an oblique orientation from camera 1. P lies within the FoV of both. State and justify assumptions; clarify all static parameters, variables, and how each is computed or determined.

**Deliverable format:** typed or scanned only, embedded in the final PDF. Camera photos of paper worksheets are explicitly not accepted.

---

## 18. Assumptions to state up front

Each of these should be one line with a justification, not just a list:

1. **Ideal pinhole projection.** Justified because both images are undistorted first using the distortion coefficients from Step 1, so the residual deviation from pinhole is below the corner-localization noise.
2. **Both cameras calibrated.** `K₁` and `K₂` known from Step 1. If the same phone is used for both views, `K₁ = K₂` — say so, and note that it halves the unknowns.
3. **Zero skew, square pixels.** Consistent with the measured `f_x ≈ f_y` from Step 1 and with the intrinsic parameter list in the slides.
4. **Rigid relative pose.** `R` and `t` are constant during capture. The rig does not flex between the two exposures.
5. **World frame anchored to camera 1.** A free choice, and the one that makes the algebra clean: camera 1's extrinsics become `[I | 0]`.
6. **P is in front of both cameras** (positive depth in both frames — the chirality condition), which is implied by "within the FoV of both."
7. **Static scene.** P does not move between the two exposures. Required if the two views are captured sequentially with one phone rather than simultaneously.

---

## 19. The derivation

### 19.1 Set up both projections

With the world frame on camera 1, using `x = K[R|t]X` from the slides:

```
x₁ ≅ K₁ [I | 0] X        →    x₁ ≅ K₁ P
x₂ ≅ K₂ [R | t] X        →    x₂ ≅ K₂ (R P + t)
```

where `P = (X, Y, Z)ᵀ` is nonhomogeneous, `R` is camera 2's rotation relative to camera 1, and `t = −R C̃` with `C̃` the camera 2 center expressed in camera 1's frame (this is the `t = −RC̃` result from the slides).

### 19.2 Back out P from camera 1

From the first equation, with `Z` the depth of P in camera 1's frame:

```
P = Z · K₁⁻¹ x₁
```

This is the same backprojection used in Step 2 — worth pointing out the connection.

### 19.3 Substitute

```
x₂ ≅ K₂ R (Z K₁⁻¹ x₁) + K₂ t
    = Z (K₂ R K₁⁻¹) x₁ + K₂ t
```

Dividing through by `Z` (legal, since the relation is projective):

```
x₂ ≅ H∞ x₁ + e₂ / Z
```

with

```
H∞ = K₂ R K₁⁻¹        (the infinite homography)
e₂ = K₂ t             (the epipole in image 2)
```

**This is the answer to the question as posed.** Read it out in words: the image of P in camera 2 is the rotation-and-intrinsics-warped position of `x₁`, displaced along the direction of the epipole by an amount inversely proportional to depth. Sweeping `Z` from 0 to ∞ traces the **epipolar line** of `x₁` in image 2 — which is exactly why a single view cannot recover depth, and why Step 2 needed a tape measure.

### 19.4 The depth-free form

Eliminating the unknown `Z` gives the epipolar constraint. Since `P₂ = RP + t` means `P₂`, `t`, and `RP` are coplanar:

```
P₂ · (t × R P) = 0
```

Substituting normalized coordinates and rearranging:

```
x₂ᵀ F x₁ = 0        with    F = K₂⁻ᵀ [t]ₓ R K₁⁻¹
```

and the essential matrix `E = [t]ₓ R = K₂ᵀ F K₁`. Worth noting the DoF count: `F` has 7, `E` has 5, and the translation is only recoverable up to scale from correspondences alone.

### 19.5 Expanded scalar form

Write out at least one row explicitly, e.g.

```
u₂ = f_x2 · (r₁ᵀP + t_x) / (r₃ᵀP + t_z) + c_x2
v₂ = f_y2 · (r₂ᵀP + t_y) / (r₃ᵀP + t_z) + c_y2
```

where `rᵢᵀ` are the rows of `R`. This makes the nonlinearity concrete and shows the perspective division explicitly. Use the ZYZ Euler parametrization of `R` from the slides (ψ, θ, φ) so the rotation is tied to three named scalars.

---

## 20. Parameter table for the report

The assignment asks explicitly for this, so make it a table:

| Symbol | Type | Meaning | How obtained |
|---|---|---|---|
| `K₁, K₂` | Static | Intrinsics of each camera | Zhang calibration, Step 1 |
| `dist₁, dist₂` | Static | Distortion coefficients | Step 1; used to undistort before applying the model |
| `R` | Static | Relative rotation, cam1 → cam2 | `cv2.stereoCalibrate` with a board visible in both views; or decompose `E` |
| `t` | Static | Relative translation, `t = −RC̃` | Same as `R`. From `E` alone it is direction-only; metric scale needs one known length (board square, or a tape-measured baseline) |
| `‖t‖` | Static | Baseline | Measured directly, or from the scale-fixing step above |
| `H∞`, `e₂`, `F`, `E` | Derived | Composite quantities | Computed from the above; `F` also estimable directly from ≥8 correspondences |
| `P = (X,Y,Z)` | Variable | Scene point | Unknown; recoverable by triangulation once `x₁, x₂, R, t` are known |
| `x₁, x₂` | Variable | Image coordinates | Measured — feature detection or manual selection |
| `Z` | Variable | Depth of P in cam 1 | Triangulated, or measured as in Step 3 |

---

## 21. Degenerate and special cases worth a paragraph

- **Pure rotation (`t = 0`).** The relation collapses to `x₂ ≅ H∞ x₁` exactly, with no depth term. Depth becomes unrecoverable and `F` is undefined. Good evidence that a nonzero baseline is essential.
- **P on a plane.** The mapping becomes a plane-induced homography, independent of depth. This connects to the coplanar-point degeneracy noted in the calibration slides.
- **P on the baseline.** Its projection lands on the epipole in both images and the constraint carries no information.

---

## 22. Optional payoff worth mentioning

Inverting the derivation gives **triangulation**: with `x₁, x₂, R, t` known, solve the two projection equations for `P` by linear least squares (`cv2.triangulatePoints`). This removes Step 2's dependence on a manually measured `Z`, and it is a strong closing paragraph — it shows the theory resolves the main limitation of the experimental part.

---

# Part E — Web Application

**As assigned:** a working demonstration as a web application, with **all assignments accessible via that webpage**, plus a screen recording of the working system. This is a submission requirement spanning the whole assignment, not a step of its own.

---

## 23. Framework choice

Both work; the deciding factor is click-to-pick-points on an image.

- **Flask + HTML canvas** — more code, full control. The image renders in a `<canvas>`, a JS click handler collects two points and POSTs them with the depth value. Recommended if the point picking needs to feel solid on video.
- **Streamlit** — much faster to write, but native click coordinates on an image need a component such as `streamlit-image-coordinates`. Fine if that dependency is acceptable.

Either way, the UI layer imports `geometry.py` and calls it. No geometry logic in the view code.

---

## 24. Pages to build

1. **Calibration (Step 1).** Upload a set of board images → run the fit → display `K`, distortion coefficients, RMS error, the per-image error table, and a few corner-overlay thumbnails. Offer `camera_params.yaml` as a download.
2. **Measurement (Step 2).** Upload an image, enter `Z` in mm, click two points → display width/height/diagonal and the annotated image.
3. **Validation (Step 3).** Load the CSV → render the measurement table, the statistics block, and the error plots.
4. **Theory (Part D).** Render the derivation. Static page or an embedded PDF is fine; the requirement is only that it is reachable from the site.

A persistent nav bar across all four, so the recording can walk through everything without retyping URLs.

---

## 25. The one real gotcha

**Canvas clicks are in display coordinates, not image coordinates.** A 4032-px-wide phone photo will be scaled down to fit the browser. If the displayed width is 800 px, every click must be multiplied by `4032/800 ≈ 5.04` before it reaches `geometry.py`, or every measurement is wrong by a factor of five.

Handle it by sending the natural image dimensions alongside the click coordinates and scaling server-side. Then verify it: measure the same object once through the CLI and once through the web app, and confirm the numbers match. Do this before recording.

---

## 26. Practical notes

- Cap upload size and reject non-images; phone photos are large.
- Compute on the **full-resolution** image even though display is scaled, since `K` was calibrated at full resolution.
- Store uploads under a gitignored directory; keep a small sample set committed so the repo is runnable from a clean clone.
- Running on `localhost` is fine for the recording. Note the run command in the README.
- The README must state the exact commands to launch the app, per the assignment's requirement for ReadMe documentation on executing the scripts.

---

## 27. Final submission checklist

- [ ] GitHub repo accessible, link included in the PDF
- [ ] ReadMe documentation at the top of each script explaining how to execute it
- [ ] Web app reaches all four parts from one page
- [ ] Screen recording of the working system captured
- [ ] Theory derivation typed or scanned, embedded in the PDF
- [ ] Error statistics for all 20 measurements reported
- [ ] PDF and video uploaded to Google Classroom
