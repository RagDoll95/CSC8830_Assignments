# Assignment 7 — Stereo measurement

Run from the repository root:

```bash
pip install -r requirements.txt
python app/app.py
# http://127.0.0.1:5000/module-7/
```

The new blueprint follows the existing assignment registry, shared template and navigation. Geometry stays in `stereo.py`; Flask only handles uploads and results. No new dependencies.

## Requirement map

| Part | Implementation / deliverable |
|---|---|
| 1: calibrated stereo object dimensions | `/module-7/`: rectangle width and height (plus opposite edges), circle diameter, every polygon edge |
| 2: uncalibrated stereo derivation | `derivation.md`: worked study notes to understand and handwrite; **not a compliant final submission** |
| 3: room dimensions from video | `/module-7/room`: one video, several stereo pairs; each room edge (width, length, height) is measured from the pair of stops that faces it |
| Submission | Accessible GitHub repo, final PDF with citations and scanned handwritten Part 2, screen recording of the working application |

The assignment's general instructions allow typed work, but Part 2 specifically says typed versions are not accepted. Follow the specific Part 2 instruction: handwrite and scan it. Do not submit these Markdown notes in place of handwriting.

## Capture and calibration

This implements **simple calibrated stereo**, not arbitrary handheld structure from motion. A single camera at two positions is a stereo setup for a static scene [1, pp. 12–13].

1. Calibrate the camera using Module 2. Preserve lens, zoom, focus, image orientation and resolution. Upload its OpenCV `camera_params.yaml`, or use the existing default only for that original phone/photo mode.
2. Put the camera on a straight guide. Take the left view, translate horizontally to the camera's right by a measured baseline B, and take the right view **without rotating or changing height**. Measure optical-center displacement, not subject distance. All points must remain visible.
3. For video, record each sideways move in one clip: stop, slide right by B, stop. Turn to face the next wall and repeat. Supply one timestamp at each stop. Disable stabilization, automatic lens switching and zoom. Calibrate using checkerboard frames from this same video mode. The default 3024×4032 photo calibration generally does not apply to video. Exact resolution matching is enforced; matching resolution alone does not guarantee matching intrinsics.
4. Upload views and B in millimetres. The app retains native image pixels; its resized display maps clicks back to those native coordinates.
5. Mark corresponding points in the same order in both images. Undo corrects a click. Download the JSON calculations after measurement.

For a rectangle, click four corners around the boundary, width edge first. For a polygon, click every vertex around its visible outline. For a circle, mark endpoints of a genuine diameter identifiable in both views; silhouette extrema can shift with viewpoint and are not automatically corresponding physical points.

For a room, measure **each edge separately**. A single view cannot show a corner together with the far ends of both floor edges: from anywhere inside a rectangular room, the two corners adjacent to any corner are at least 90° apart (the room is inscribed in the circle whose diameter joins them), wider than a phone's main camera. So, for each edge, choose Width, Length or Height, enter the timestamps of the two stops facing it (and the baseline of that move), then click the edge's two endpoints in both frames — for example the two floor corners of a wall, or the floor and ceiling ends of a vertical corner. Repeat for the other edges; measuring an edge again from another pair is encouraged. The page lists every measured edge, lets you remove a bad one, and reports each dimension's mean and the spread between repeats. This measures the edges you select; it is not an automatic floor-plan extraction system. If furniture hides an endpoint, choose a clearer view rather than substitute unrelated points.

## Method

Undistort each selected point with the existing Module 2 intrinsics and distortion coefficients. The lecture's simple-stereo model ignores lens distortion [1, p. 11]; undistortion is an addition, done with OpenCV [3], so that the pinhole equations below hold for a real phone lens. With normalized rays `(xL,yL)` and `(xR,yR)`:

- disparity in corrected pixel units: `d = fx * (xL - xR)`;
- depth first: `Z = fx * B / d`;
- position: `X = xL * Z`, `Y = yL * Z`;
- dimension: `sqrt((Xj-Xi)^2 + (Yj-Yi)^2 + (Zj-Zi)^2)`.

Each endpoint has its own depth. There is no supplied object distance and no same-depth assumption. Z is axial depth, not radial distance to the camera. The camera frame uses X right, Y down, Z forward. See [1, pp. 12–13]. Correspondences are manually marked; dense matching is unnecessary for a small set of requested edges.

Nonpositive/near-zero disparity, invalid points, out-of-image points, calibration-size mismatches and vertical disagreement above 3 corrected pixels are rejected. Disparities below 10 pixels trigger a warning. Vertical agreement is necessary but insufficient to validate camera pose. Incorrect horizontal rotation, calibration, baseline or correspondences can still bias measurements. Near-zero disparity is particularly sensitive: approximately `|δZ|/Z ≈ |δd|/d` if f and B are fixed. No physical accuracy claim is made without ruler/tape comparisons.

## Reproducible examples and checks

```bash
python week_7/module7/demo.py --out /tmp/module7-demo
python -m unittest discover -s week_7/module7 -p 'test_*.py' -v
python week_7/module7/stereo.py \
  --camera /tmp/module7-demo/camera.yaml \
  --points /tmp/module7-demo/object_points.json \
  --baseline-mm 200 --shape rectangle
```

Upload generated `camera.yaml` and use B=200 mm. Object inputs are `left.png` and `right.png`. Room input is `room.avi`: stops 0.5 s and 1.5 s face the far wall (measure Width and Height), stops 2.5 s and 3.5 s face the side wall (measure Length). Click the numbered cross centers in matching order. Exact coordinates are in `points.json` for numerical tests. These images and video are **synthetic**, clearly labelled, and are not evidence of performance in a real house.

Expected synthetic object: 1000×600 mm, Z=3000 mm. Expected room, viewed from inside: width 3000, length 4000, height 2400 mm. Automated tests check all shape modes, distortion correction, invalid geometry, frame selection by timestamp, and both Flask upload/measurement/download flows. Manual clicks on rounded demo pixels introduce small deviations.

## Screen recording and final report

Record the application loading your calibrated views, marking points, calculating Z and dimensions, and downloading results. Demonstrate all object shape cases and the room-video flow. Include the capture setup, baseline, calibration resolution, timestamps, measured dimensions, tape/ruler ground truth and absolute/percentage errors. The synthetic example is useful for explaining the algorithm, but supplement it with real captures for the requested house-room application.

No real room footage, stereo capture baseline, ground-truth dimensions or handwritten pages were supplied with this assignment. Those remain necessary for a completed empirical report and submission recording. Do not fabricate them.

## References

1. Shree K. Nayar. *Camera Calibration*, FPCV-4-1, April 2025, especially pp. 12–13. https://cave.cs.columbia.edu/Statics/monographs/Camera%20Calibration%20FPCV-4-1.pdf
2. Shree K. Nayar. *Uncalibrated Stereo*, FPCV-4-2, April 2025, especially pp. 2–14. https://cave.cs.columbia.edu/Statics/monographs/Uncalibrated%20Stereo%20FPCV-4-2.pdf
3. OpenCV. *Camera Calibration and 3D Reconstruction*, `undistortPoints`. https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
