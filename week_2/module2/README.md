# Module 2 — Camera Calibration and Real-World Measurement

**CSc 8830: Computer Vision** · Python + OpenCV + Flask

Camera calibration by Zhang's multi-plane method, real-world 2D measurement by inverting
the perspective projection at a known object distance, a 20-measurement validation
experiment with error statistics, the two-camera derivation, and a web application that
reaches all four parts from one page.

---
## Install

```bash
cd week_2/module2
pip install -r requirements.txt
```

Verified on Python 3.9.23 with opencv-python 5.0.0, numpy 2.0.2, Flask 3.1.3,
matplotlib 3.9.4.

---

## Launch the web application

```bash
cd ../..            # the repository root
python app/app.py
# open http://127.0.0.1:5000        -> every assignment
#      http://127.0.0.1:5000/module-2/   -> this one
```

Options: `--port 8000`, `--host 0.0.0.0`, `--debug`.


| Page | Part | What it does |
|---|---|---|
| `/` | — | Home: every assignment in the repository |
| `/module-2/` | A · Step 1 | Calibration: upload board images → fit `K` and distortion → per-image error table → corner overlays → download `camera_params.yaml` |
| `/module-2/measure` | B · Step 2 | Upload an image, enter `Z`, click two points → width/height/diagonal + error budget + annotated image |
| `/module-2/records` | C · Step 3 | Measurement table, error statistics, diagnostic plots, report figures `.zip` |
---

