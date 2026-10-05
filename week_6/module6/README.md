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

Use local video files; OpenCV cannot read YouTube watch-page URLs. Run each video
separately. Compare corresponding corners manually in the two exported frames to
validate predicted positions. The derivation and bilinear interpolation are in
`theory.md`. Final PDF, screen recording and Part B's four-view example remain to
be completed with actual footage/images and measurements.

References: Nayar (2025), *Optical Flow*, FPCV-4-3; *Structure from Motion*, FPCV-4-4.
OpenCV: https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
