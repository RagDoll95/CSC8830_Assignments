"""Run: python -m unittest discover -s week_7/module7 -p 'test_*.py' -v.

Numerical ground truth, distortion correction, failure cases and Flask photo /
video workflows. Synthetic data verify implementation, not physical accuracy.
"""
import io
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import demo
import stereo


class StereoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.points = demo.generate(self.folder)
        self.cam = stereo.camera(self.folder / 'camera.yaml')

    def tearDown(self):
        self.temp.cleanup()

    def test_shapes_and_depth(self):
        p = self.points['object']
        result = stereo.measure(**p, cam=self.cam, baseline_mm=200, shape='rectangle')
        np.testing.assert_allclose(result['points_mm'], demo.OBJECT, atol=1e-8)
        np.testing.assert_allclose([d['mm'] for d in result['dimensions']], [1000, 600, 1000, 600])
        for shape in ('circle', 'polygon'):
            n = 2 if shape == 'circle' else 4
            result = stereo.measure(p['left'][:n], p['right'][:n], self.cam, 200, shape)
            self.assertAlmostEqual(result['dimensions'][0]['mm'], 1000)
        result = stereo.measure(**self.points['room'], cam=self.cam, baseline_mm=200, shape='room')
        np.testing.assert_allclose([d['mm'] for d in result['dimensions']], [3000, 2000, 2400])
        self.assertEqual(result['warnings'], [])

    def test_distorted_points(self):
        self.cam.dist = np.array([[.1, -.02, .001, -.001, 0.]])
        views = [cv2.projectPoints(demo.OBJECT, np.zeros(3), np.array([-b, 0., 0.]), demo.K, self.cam.dist)[0].reshape(-1, 2) for b in (0., 200.)]
        result = stereo.measure(*views, self.cam, 200, 'rectangle')
        np.testing.assert_allclose(result['points_mm'], demo.OBJECT, atol=.01)

    def test_bad_geometry(self):
        p = self.points['object']
        for left, right, baseline in [(p['right'], p['left'], 200), (p['left'], p['left'], 200), (p['left'], p['right'], -1), (p['left'], p['right'], float('nan'))]:
            with self.assertRaises(ValueError):
                stereo.measure(left, right, self.cam, baseline, 'rectangle')
        right = np.array(p['right']); right[:, 1] += 4
        with self.assertRaisesRegex(ValueError, 'Vertical'):
            stereo.measure(p['left'], right, self.cam, 200, 'rectangle')
        with self.assertRaises(ValueError):
            stereo.frame_at(self.folder / 'room.avi', 99)

    def test_web_photo_and_video(self):
        sys.path.insert(0, str(stereo.ROOT / 'app'))
        from app import create_app
        from modules import module7
        app = create_app(); app.testing = True
        self.assertIn('module7', app.mounted)
        client = app.test_client()
        self.assertEqual(client.get('/module-7/').status_code, 200)
        self.assertIn(b'Module 7', client.get('/').data)
        for room in (False, True):
            data = {'baseline': '200', 'camera': (io.BytesIO((self.folder / 'camera.yaml').read_bytes()), 'camera.yaml')}
            route = '/module-7/room' if room else '/module-7/'
            if room:
                data.update(video=(io.BytesIO((self.folder / 'room.avi').read_bytes()), 'room.avi'), left_time='0', right_time='2')
            else:
                data.update(shape='rectangle')
                for side in ('left', 'right'):
                    data[side] = (io.BytesIO((self.folder / (side + '.png')).read_bytes()), side + '.png')
            response = client.post(route, data=data)
            self.assertEqual(response.status_code, 200, response.text)
            run = re.search(r'name="run" value="([a-f0-9]+)"', response.text).group(1)
            try:
                response = client.post(route, data={'action': 'measure', 'run': run, 'points': json.dumps(self.points['room' if room else 'object'])})
                self.assertEqual(response.status_code, 200, response.text)
                with client.get(f'/module-7/output/{run}/result.json') as response:
                    result = response.json
                self.assertAlmostEqual(result['dimensions'][0]['mm'], 3000 if room else 1000)
                self.assertEqual(client.get(f'/module-7/output/{run}/camera.yaml').status_code, 404)
            finally:
                shutil.rmtree(module7.run_folder(run))
        self.assertEqual(client.post('/module-7/', data={}).status_code, 400)
        self.assertEqual(client.get('/module-7/output/invalid/left.png').status_code, 404)


if __name__ == '__main__':
    unittest.main()
