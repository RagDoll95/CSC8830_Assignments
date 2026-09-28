# Module 4 — RGB and thermal human boundary detection

## Run the web demonstration
From the repository root:
```bash
pip install -r requirements.txt
python app/app.py
```
Open http://127.0.0.1:5000/module-4/ and upload an RGB or thermal image.
All assignments remain available from the home page and shared navigation.
There are no sliders or parameter controls. Download any displayed stage as PNG.

## Run the same pipeline from the command line
```bash
python week_4/module4/boundaries.py person.jpg --output rgb_results
python week_4/module4/boundaries.py thermal.tiff --output thermal_results
python -m unittest discover -s week_4/module4 -p 'test_*.py'
```
Each script starts with its execution instructions. The CLI writes original.png,
edges.png, closed.png, and overlay.png. The web app uses ignored session uploads.

## Fixed OpenCV pipeline
1. Convert to grayscale with `cv2.cvtColor`.
2. Smooth using `cv2.GaussianBlur`: 5×5 kernel, sigma=0 (OpenCV derives sigma).
3. Detect edges with `cv2.Canny`: thresholds 50/150, aperture 3, L2 gradient.
4. Bridge small gaps with `cv2.morphologyEx`: CLOSE, 3×3 ellipse, one iteration.
5. Extract external contours with `cv2.findContours`: RETR_EXTERNAL and CHAIN_APPROX_SIMPLE.
6. Draw all candidate contours in green with `cv2.drawContours`.

16-bit images are scaled
from their observed minimum/maximum to 0–255 because Canny needs 8-bit input.
Grayscale thermal images are preferred. False-color thermal RGB images are
accepted, but grayscale conversion of palette colors is not a temperature map.

## Interpretation and limitations
This is a Canny/contour baseline, so external contour retrieval does
not guarantee that the contour corresponds to a closed human silhouette.
We deliberately do not assume that the largest contour represents the person.
No exact-boundary accuracy claim is made without evaluation on actual images.

Repository: https://github.com/RagDoll95/CSC8830_Assignments
