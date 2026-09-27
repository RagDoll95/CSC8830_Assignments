"""Module 4: classical boundary extraction with OpenCV (no learned models).

Run from the repository root:
    python week_4/module4/boundaries.py person.jpg --output module4_output
    python week_4/module4/boundaries.py thermal.tiff --output thermal_output

Accepts 8-bit grayscale/BGR/BGRA and 16-bit grayscale images. Writes the input
preview, Canny edges, closed edges, and candidate contour overlay as PNGs.
Both the CLI and Flask application use detect_boundaries() below.
"""
from pathlib import Path
import argparse

import cv2
import numpy as np


def detect_boundaries(image):
    """Return display stages and contour count using fixed parameters.

    Contours are candidate intensity boundaries, not semantic person labels.
    Keep all external contours rather than assume the largest one is a person.
    """
    if image is None or image.size == 0:
        raise ValueError('The image could not be read.')
    if image.dtype not in (np.uint8, np.uint16):
        raise ValueError('Use an 8-bit or 16-bit PNG, TIFF, JPG, or BMP image.')
    if image.ndim == 2:
        gray = image
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        color = image[:, :, :3]
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError('Use a grayscale, RGB, or RGBA image.')
    # Canny requires 8-bit input. Thermal sensor images may be 16-bit;
    # scale their observed intensity range without interpreting temperature.
    if image.dtype == np.uint16:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        color = cv2.normalize(color, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3, L2gradient=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = color.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return {'original': color, 'edges': edges, 'closed': closed, 'overlay': overlay}, len(contours)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image')
    parser.add_argument('--output', default='module4_output')
    args = parser.parse_args()
    try:
        stages, count = detect_boundaries(cv2.imread(args.image, cv2.IMREAD_UNCHANGED))
    except ValueError as exc:
        parser.error(str(exc))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, stage in stages.items():
        if not cv2.imwrite(str(output / (name + '.png')), stage):
            raise OSError('Could not write output image.')
    print(f'Saved {count} candidate contours and image stages to {output}')


if __name__ == '__main__':
    main()
