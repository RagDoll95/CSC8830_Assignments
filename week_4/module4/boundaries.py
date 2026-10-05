"""

Run from the repository root:
    python week_4/module4/boundaries.py person.jpg --output module4_output
    python week_4/module4/boundaries.py thermal.tiff --output thermal_output

Accepts 8-bit grayscale/BGR/BGRA and 16-bit grayscale images. Writes the input
preview, Canny edges, closed edges, and foreground outline as PNGs.
Assumes the upper-left 20x20 pixels represent a roughly uniform background.
Both the CLI and Flask application use detect_boundaries() below.
"""
from pathlib import Path
import argparse

import cv2
import numpy as np


def detect_boundaries(image):
    """Return display stages and contour count using fixed parameters.

    Separate a uniform background by color and outline the largest remaining
    component. This assumes one dominant subject and a background-only corner;
    it is not a general person detector. Canny stages are kept for comparison.
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
    # Sample the background in BGR, before grayscale conversion discards color.
    # Assumption: this corner contains background, and its color is fairly uniform.
    sample = color[:20, :20].reshape(-1, 3)
    background_color = np.median(sample, axis=0).astype(np.int16)
    # Signed arithmetic avoids uint8 wraparound near 0 and 255.
    lower = np.clip(background_color - 15, 0, 255).astype(np.uint8)
    upper = np.clip(background_color + 15, 0, 255).astype(np.uint8)
    background_mask = cv2.inRange(color, lower, upper)
    foreground_mask = cv2.bitwise_not(background_mask)

    # Remove detached foreground specks; assume the largest region is the subject.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground_mask, connectivity=8
    )
    mask = np.zeros(gray.shape, dtype=np.uint8)
    if count > 1:  # Label 0 is background; a uniform image has no foreground.
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask[labels == largest] = 255
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
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
