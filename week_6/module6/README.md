# Assignment 6A: optical flow

```bash
pip install -r requirements.txt
python app/app.py
# Open http://127.0.0.1:5000/module-6/
```

Choose the bundled penguin or marble-race sample (30 seconds each), or upload a
video and select a start time with at least 30 seconds remaining.
Or run directly:

```bash
python week_6/module6/motion.py video.mp4 --out output --start 0
```

The script detects corners and uses OpenCV's pyramidal Lucas-Kanade method from the
Optical Flow slides. Yellow arrows show displacement between consecutive frames.
Corners are redetected each frame; these are image-motion vectors, not object identities.
Frames retain the source FPS and are resized to at most 640 pixels wide.

Outputs: flow video, point coordinates/displacements in `points.csv`, and the first two
frames with a flow overlay. CSV rows with `frame=0` correspond to that pair. Coordinates
use the resized images. Install ffmpeg optionally for MP4 output; otherwise play AVI
locally. The web upload limit is 200 MB. Use the script if web processing times out.

Chosen videos:
- Penguin walk: https://www.youtube.com/watch?v=fdk7X4LrgS4 — sample 00:30–01:00.
- Marble race: https://www.youtube.com/watch?v=0sHu99vdz-A — sample 02:00–02:30.

Samples are silent, 640-pixel-wide MP4s retaining the source frame rates.

## Validation and four-view example

Bundled samples include pixel validation against approximate visual annotations in
`measurements.csv`. Uploaded videos have no supplied annotations. The two-frame
figures, errors, and normal-equation matrices are downloadable in the app.

The **Run four-view example** button reconstructs a simulated planar L-shaped card.
It uses the slides' centering and SVD with rank 2, a known front-parallel first camera,
and a scale of 70 pixels/cm. It exports four images, camera settings and positions,
all calculation matrices, and the reconstructed boundary. This is not captured data
or general uncalibrated 3D reconstruction. See `report.pdf` for the assumptions.

Reproduce the report figures and results from the repository root:

```bash
python week_6/module6/validate.py penguins --out week_6/module6/output/penguins
python week_6/module6/validate.py marble_race --out week_6/module6/output/marble_race
python week_6/module6/sfm.py --out week_6/module6/output/structure
cd week_6/module6
pdflatex report.tex
```

For measured four-view coordinates, use `sfm.py --input data.json --out output`.
The JSON requires `points` (4 views × N ordered boundary points × [x,y]) and
`scale` (pixels per object unit); the first view must be front-parallel and all views
must share the scale. Supply the actual photographs and camera information alongside
that data. The simulated camera positions are inputs, not recovered distances.

`report.pdf` includes calculations, figures, sources, and limitations. A recording
of the running app is still needed for the classroom submission. If captured images
are required for Part B, replace the explicitly simulated example with your four views.

References: Nayar (2025), *Optical Flow*, FPCV-4-3; *Structure from Motion*, FPCV-4-4.
