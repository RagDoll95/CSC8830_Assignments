# Assignment 6: optical flow, tracking and structure from motion

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

## Two-frame tracking check

Bundled samples are also checked against hand-measured pixel locations in their first
two frames (`measurements.csv`): pyramidal Lucas-Kanade predictions, SSD template
matching, and the single-step normal equations with the slides' 2x2x2-cube derivatives.
`flow_summary.png` separates the dominant (camera) flow from the rest.

## Four-view structure from motion

`sfm.py` is Tomasi-Kanade factorization (FPCV-4-4) for a flat object: centroid-subtracted
observation matrix W, rank-2 SVD (a plane), and the 2x2 matrix Q from the orthonormality
of every camera's i, j by Newton's method. Points are undistorted and normalized with the
Module 2 calibration (`week_2/module2/calibration/output/camera_params.yaml`).

The default uses four Module 2 checkerboard photos taken from about 1.05 m
(IMG_3930, 3917, 3928, 3938). The known 24.9 mm grid is used only for one scale distance
and for grading. The app page `/module-6/structure` also takes four new photos, with
corners detected (checkerboard) or clicked by hand in the same order in every photo.
Use the calibrated phone at 1x, portrait, from the same distance in every view.

## Reproduce the write-up

```bash
python week_6/module6/validate.py penguins --out week_6/module6/output/penguins
python week_6/module6/validate.py marble_race --out week_6/module6/output/marble_race
python week_6/module6/sfm.py --out week_6/module6/output/structure
cd week_6/module6 && pdflatex answers.tex && pdflatex answers.tex
```

For your own photos from the command line:
`sfm.py --images a.jpg b.jpg c.jpg d.jpg --points points.json --out output`, where
`points.json` is `{"points": [4 views x N x [x, y]], "known": [i, j, mm]}`.

`answers.pdf` is the submission write-up. Hosted app: https://ragman95.pythonanywhere.com/

References: Nayar (2025), *Optical Flow* FPCV-4-3, *Structure from Motion* FPCV-4-4,
*Object Tracking* FPCV-5-1; Tomasi and Kanade (1992).
