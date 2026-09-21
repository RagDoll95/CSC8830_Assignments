"""Run with: python -m unittest week_3/module3/test_image_blur.py"""

import importlib.util
import pathlib
import sys
import unittest

import numpy as np

PATH = pathlib.Path(__file__).with_name("image_blur.py")
SPEC = importlib.util.spec_from_file_location("image_blur", PATH)
blur = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = blur
SPEC.loader.exec_module(blur)


class ImageBlurTests(unittest.TestCase):
    def check_equivalence(self, image, kind, size, sigma=2.0):
        kernel = blur.make_kernel(kind, size, sigma)
        spatial = blur.spatial_convolution(image, kernel)
        fourier = blur.fourier_convolution(image, kernel)
        self.assertTrue(np.allclose(spatial, fourier, atol=1e-9))

    def test_grayscale_box(self):
        image = np.arange(99, dtype=np.uint8).reshape(9, 11)
        self.check_equivalence(image, "box", 5)

    def test_color_gaussian(self):
        rng = np.random.default_rng(8830)
        image = rng.integers(0, 256, (13, 17, 3), dtype=np.uint8)
        self.check_equivalence(image, "gaussian", 7, 1.8)

    def test_kernel_is_normalized(self):
        self.assertAlmostEqual(blur.make_kernel("gaussian", 9, 2.0).sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
