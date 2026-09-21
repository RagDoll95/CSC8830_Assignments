"""Module 3: image blurring in the spatial and Fourier domains.

Run from the repository root:
    python week_3/module3/image_blur.py input.jpg --filter gaussian --size 9

The web application imports the same functions from this file.  Both paths use
the same normalized, symmetric kernel and zero values beyond the image border.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Comparison:
    average_absolute_difference: float
    maximum_absolute_difference: float


def make_kernel(kind: str, size: int, sigma: float = 2.0) -> np.ndarray:
    """Return a normalized odd-sized box or Gaussian kernel."""
    if size < 3 or size > 31 or size % 2 == 0:
        raise ValueError("kernel size must be an odd number from 3 to 31")
    if kind == "box":
        kernel = np.ones((size, size), dtype=np.float64)
    elif kind == "gaussian":
        if sigma <= 0:
            raise ValueError("sigma must be greater than zero")
        axis = np.arange(size, dtype=np.float64) - size // 2
        xx, yy = np.meshgrid(axis, axis)
        kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    else:
        raise ValueError("filter must be 'box' or 'gaussian'")
    return kernel / kernel.sum()


def spatial_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve a grayscale or color image directly in the spatial domain."""
    source = _as_channels(image).astype(np.float64)
    kh, kw = kernel.shape
    py, px = kh // 2, kw // 2
    padded = np.pad(source, ((py, py), (px, px), (0, 0)), mode="constant")
    output = np.zeros_like(source, dtype=np.float64)
    # The kernel is symmetric, but it is flipped explicitly to implement
    # convolution rather than correlation.
    flipped = kernel[::-1, ::-1]
    for row in range(kh):
        for col in range(kw):
            output += flipped[row, col] * padded[
                row:row + source.shape[0], col:col + source.shape[1], :
            ]
    return _restore_channels(output, image)


def fourier_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve by multiplying Fourier transforms, then taking the inverse."""
    source = _as_channels(image).astype(np.float64)
    h, w, channels = source.shape
    kh, kw = kernel.shape
    full_shape = (h + kh - 1, w + kw - 1)
    filter_transform = np.fft.fft2(kernel, s=full_shape)
    full = np.empty((full_shape[0], full_shape[1], channels), np.float64)
    for channel in range(channels):
        image_transform = np.fft.fft2(source[:, :, channel], s=full_shape)
        full[:, :, channel] = np.fft.ifft2(
            image_transform * filter_transform
        ).real
    py, px = kh // 2, kw // 2
    same = full[py:py + h, px:px + w, :]
    return _restore_channels(same, image)


def compare(spatial: np.ndarray, fourier: np.ndarray) -> Comparison:
    difference = np.abs(spatial.astype(np.float64) - fourier.astype(np.float64))
    return Comparison(
        average_absolute_difference=float(difference.mean()),
        maximum_absolute_difference=float(difference.max()),
    )


def display_image(image: np.ndarray) -> np.ndarray:
    """Clip floating-point convolution output for an ordinary image file."""
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def difference_image(spatial: np.ndarray, fourier: np.ndarray) -> np.ndarray:
    """Scale the absolute difference so very small numerical values are visible."""
    difference = np.abs(spatial.astype(np.float64) - fourier.astype(np.float64))
    maximum = difference.max()
    if maximum == 0:
        return np.zeros_like(display_image(difference))
    return np.clip(np.rint(difference * 255.0 / maximum), 0, 255).astype(np.uint8)


def process(image: np.ndarray, kind: str, size: int, sigma: float = 2.0):
    kernel = make_kernel(kind, size, sigma)
    spatial = spatial_convolution(image, kernel)
    fourier = fourier_convolution(image, kernel)
    return spatial, fourier, difference_image(spatial, fourier), compare(spatial, fourier)


def _as_channels(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image[:, :, None]
    if image.ndim == 3:
        return image
    raise ValueError("image must be grayscale or color")


def _restore_channels(image: np.ndarray, original: np.ndarray) -> np.ndarray:
    return image[:, :, 0] if original.ndim == 2 else image


def main(argv=None) -> int:
    import cv2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--filter", choices=("box", "gaussian"), default="gaussian")
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--output", default="module3_output")
    args = parser.parse_args(argv)

    image = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if image is None:
        parser.error("could not read image: %s" % args.image)
    spatial, fourier, difference, metrics = process(
        image, args.filter, args.size, args.sigma
    )
    os.makedirs(args.output, exist_ok=True)
    cv2.imwrite(os.path.join(args.output, "spatial.png"), display_image(spatial))
    cv2.imwrite(os.path.join(args.output, "fourier.png"), display_image(fourier))
    cv2.imwrite(os.path.join(args.output, "difference.png"), difference)
    with open(os.path.join(args.output, "comparison.json"), "w") as stream:
        json.dump(asdict(metrics), stream, indent=2)
    print(json.dumps(asdict(metrics), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
