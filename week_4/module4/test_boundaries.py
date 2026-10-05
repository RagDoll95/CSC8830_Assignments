"""Run: python -m unittest discover -s week_4/module4 -p 'test_*.py'."""
import io
import sys
import unittest
from pathlib import Path
import cv2
import numpy as np
from boundaries import detect_boundaries


class BoundaryTests(unittest.TestCase):
    def test_rectangle_boundary_and_input_preservation(self):
        image = np.zeros((100, 100, 3), np.uint8)
        image[20:80, 30:70] = 255
        before = image.copy()
        stages, count = detect_boundaries(image)
        self.assertEqual(count, 1)
        self.assertTrue(np.array_equal(image, before))
        ys, xs = np.nonzero(stages['edges'])
        self.assertLessEqual(abs(xs.min() - 30), 2)
        self.assertLessEqual(abs(xs.max() - 69), 2)
        self.assertLessEqual(abs(ys.min() - 20), 2)
        self.assertLessEqual(abs(ys.max() - 79), 2)

    def test_thermal_and_uniform(self):
        image = np.full((100, 100), 10000, np.uint16)
        self.assertEqual(detect_boundaries(image)[1], 0)
        image[20:80, 30:70] = 20000
        stages, count = detect_boundaries(image)
        self.assertEqual(count, 1)
        self.assertEqual(stages['edges'].dtype, np.uint8)

    def test_web_routes_uploads_and_bad_files(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'app'))
        from app import create_app
        app = create_app()
        self.assertEqual(app.broken, [])
        client = app.test_client()
        for route in ['/', '/module-2/', '/module-3/', '/module-4/']:
            self.assertEqual(client.get(route).status_code, 200)
        for dtype in [np.uint8, np.uint16]:
            image = np.zeros((100, 100), dtype)
            image[20:80, 30:70] = np.iinfo(dtype).max
            _, data = cv2.imencode('.png', image)
            response = client.post('/module-4/', data={'image': (io.BytesIO(data.tobytes()), 'person.png')})
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'1 candidate contour', response.data)
            import re
            paths = re.findall(rb'src="(/module-4/output/[^\"]+)"', response.data)
            self.assertEqual(len(paths), 4)
            for path in paths:
                with client.get(path.decode()) as download:
                    self.assertEqual(download.status_code, 200)
        self.assertEqual(client.post('/module-4/', data={}).status_code, 400)
        self.assertEqual(client.post('/module-4/', data={'image': (io.BytesIO(b'bad'), 'bad.png')}).status_code, 400)


if __name__ == '__main__':
    unittest.main()
