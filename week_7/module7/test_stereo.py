"""Run: python -m unittest discover -s week_7/module7 -p 'test_*.py' -v.

Numerical ground truth, distortion correction, handheld pose estimation,
failure cases and Flask photo / video workflows. Synthetic data verify implementation, not physical accuracy.
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

sys.path.insert(0, str(stereo.ROOT / 'app'))
from app import create_app  # noqa: E402
from modules import module7  # noqa: E402


class SyntheticCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.points = demo.generate(self.folder)
        self.cam = stereo.camera(self.folder / 'camera.yaml')

    def tearDown(self):
        self.temp.cleanup()

    def room_edge(self, label):
        return next(edge for edge in self.points['room'] if edge['label'] == label)


class MeasureTests(SyntheticCase):
    def test_rectangle_recovers_each_corner_depth_and_position(self):
        p = self.points['object']

        result = stereo.measure(p['left'], p['right'], self.cam, 200, 'rectangle')

        np.testing.assert_allclose(result['points_mm'], demo.OBJECT, atol=1e-8)

    def test_rectangle_reports_width_height_and_opposite_sides(self):
        p = self.points['object']

        result = stereo.measure(p['left'], p['right'], self.cam, 200, 'rectangle')

        np.testing.assert_allclose([d['mm'] for d in result['dimensions']], [1000, 600, 1000, 600])

    def test_circle_reports_one_diameter(self):
        p = self.points['object']

        result = stereo.measure(p['left'][:2], p['right'][:2], self.cam, 200, 'circle')

        self.assertEqual([(d['label'], round(d['mm'], 6)) for d in result['dimensions']], [('Diameter', 1000.0)])

    def test_polygon_reports_every_edge(self):
        p = self.points['object']

        result = stereo.measure(p['left'], p['right'], self.cam, 200, 'polygon')

        self.assertEqual([d['label'] for d in result['dimensions']], ['Edge 1', 'Edge 2', 'Edge 3', 'Edge 4'])

    def test_edge_reports_one_distance(self):
        p = self.points['object']

        result = stereo.measure(p['left'][:2], p['right'][:2], self.cam, 200, 'edge')

        self.assertEqual([(d['label'], round(d['mm'], 6)) for d in result['dimensions']], [('Distance', 1000.0)])

    def test_edge_rejects_more_than_two_points(self):
        p = self.points['object']

        with self.assertRaisesRegex(ValueError, 'edge: 2 endpoints'):
            stereo.measure(p['left'][:3], p['right'][:3], self.cam, 200, 'edge')

    def test_distorted_clicks_are_undistorted_before_triangulation(self):
        self.cam.dist = np.array([[.1, -.02, .001, -.001, 0.]])
        views = [cv2.projectPoints(demo.OBJECT, np.zeros(3), np.array([-b, 0., 0.]), demo.K, self.cam.dist)[0].reshape(-1, 2)
                 for b in (0., 200.)]

        result = stereo.measure(*views, self.cam, 200, 'rectangle')

        np.testing.assert_allclose(result['points_mm'], demo.OBJECT, atol=.01)

    def test_swapped_views_are_rejected_as_negative_disparity(self):
        p = self.points['object']

        with self.assertRaisesRegex(ValueError, 'Disparity must exceed 1 pixel'):
            stereo.measure(p['right'], p['left'], self.cam, 200, 'rectangle')

    def test_identical_views_are_rejected_as_zero_disparity(self):
        p = self.points['object']

        with self.assertRaisesRegex(ValueError, 'Disparity must exceed 1 pixel'):
            stereo.measure(p['left'], p['left'], self.cam, 200, 'rectangle')

    def test_negative_baseline_is_rejected(self):
        p = self.points['object']

        with self.assertRaisesRegex(ValueError, 'Baseline must be a positive'):
            stereo.measure(p['left'], p['right'], self.cam, -1, 'rectangle')

    def test_nan_baseline_is_rejected(self):
        p = self.points['object']

        with self.assertRaisesRegex(ValueError, 'Baseline must be a positive'):
            stereo.measure(p['left'], p['right'], self.cam, float('nan'), 'rectangle')

    def test_vertical_mismatch_over_three_pixels_warns(self):
        p = self.points['object']
        right = np.array(p['right'])
        right[:, 1] += 4

        result = stereo.measure(p['left'], right, self.cam, 200, 'rectangle')

        self.assertIn('Vertical mismatch exceeds 3 pixels', ' '.join(result['warnings']))

    def test_vertical_mismatch_over_three_pixels_still_estimates_dimensions(self):
        p = self.points['object']
        right = np.array(p['right'])
        right[:, 1] += 4

        result = stereo.measure(p['left'], right, self.cam, 200, 'rectangle')

        np.testing.assert_allclose([d['mm'] for d in result['dimensions']], [1000, 600, 1000, 600], rtol=.02)

    def test_frame_after_end_of_video_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside the video'):
            stereo.frame_at(self.folder / 'room.avi', 99)


def true_pose(times_s):
    """Exact pose of the second synthetic handheld frame relative to the first."""
    (r1, c1), (r2, c2) = [demo.handheld_view(t) for t in times_s]
    translation = r2 @ (c1 - c2)
    return stereo.Pose(r2 @ r1.T, translation / np.linalg.norm(translation), 100, 5.)


class HandheldTests(SyntheticCase):
    def frames(self, edge):
        return [stereo.frame_at(self.folder / 'room.avi', t) for t in edge['times_s']]

    def measure(self, edge, pose=None, left=None):
        pose = pose or true_pose(edge['times_s'])
        return stereo.measure_handheld(left or edge['left'], edge['right'], self.cam, pose, edge['reference_mm'])

    def test_known_pose_recovers_exact_edge_length(self):
        edge = self.room_edge('Length')

        result = self.measure(edge)

        self.assertAlmostEqual(result['mm'], demo.ROOM_SIZE['Length'], places=6)

    def test_known_pose_recovers_metric_camera_move(self):
        edge = self.room_edge('Width')
        (_, c1), (_, c2) = [demo.handheld_view(t) for t in edge['times_s']]

        result = self.measure(edge)

        self.assertAlmostEqual(result['camera_move_mm'], np.linalg.norm(c2 - c1), places=6)

    def test_pose_from_video_frames_matches_true_rotation(self):
        edge = self.room_edge('Length')
        truth = true_pose(edge['times_s'])

        pose = stereo.relative_pose(*self.frames(edge), self.cam)

        angle = np.degrees(np.arccos(np.clip((np.trace(pose.rotation @ truth.rotation.T) - 1) / 2, -1, 1)))
        self.assertLess(angle, .5)

    def test_pose_from_video_frames_matches_true_direction_of_travel(self):
        edge = self.room_edge('Length')
        truth = true_pose(edge['times_s'])

        pose = stereo.relative_pose(*self.frames(edge), self.cam)

        self.assertLess(np.degrees(np.arccos(np.clip(pose.translation @ truth.translation, -1, 1))), 2)

    def test_estimated_pose_measures_each_room_edge_within_two_percent(self):
        lengths = {edge['label']: self.measure(edge, stereo.relative_pose(*self.frames(edge), self.cam))['mm']
                   for edge in self.points['room']}

        np.testing.assert_allclose([lengths[k] for k in demo.ROOM_SIZE], list(demo.ROOM_SIZE.values()), rtol=.02)

    def test_camera_that_only_turns_is_rejected(self):
        view = demo.handheld_view(.3)
        turn = cv2.Rodrigues(np.array([0., .05, 0.]))[0]
        frames = [demo.render_room(view, 0, 'still'), demo.render_room((turn @ view[0], view[1]), 0, 'turned')]

        with self.assertRaisesRegex(ValueError, 'stepped sideways'):
            stereo.relative_pose(*frames, self.cam)

    def test_blank_frames_are_rejected_for_lack_of_texture(self):
        blank = np.full((demo.SIZE[1], demo.SIZE[0], 3), 200, np.uint8)

        with self.assertRaisesRegex(ValueError, 'Too little texture'):
            stereo.relative_pose(blank, blank, self.cam)

    def test_click_off_by_ten_pixels_warns(self):
        edge = self.room_edge('Width')
        left = np.array(edge['left'])
        left[3, 1] += 10

        result = self.measure(edge, left=left.tolist())

        self.assertIn('more than 3 pixels', ' '.join(result['warnings']))

    def test_click_off_by_ten_pixels_still_estimates_length(self):
        edge = self.room_edge('Width')
        left = np.array(edge['left'])
        left[3, 1] += 10

        result = self.measure(edge, left=left.tolist())

        self.assertAlmostEqual(result['mm'], demo.ROOM_SIZE['Width'], delta=150)

    def test_exact_clicks_have_no_warnings(self):
        self.assertEqual(self.measure(self.room_edge('Height'))['warnings'], [])

    def test_two_points_instead_of_four_are_rejected(self):
        edge = self.room_edge('Width')

        with self.assertRaisesRegex(ValueError, 'Mark 4 matching points'):
            stereo.measure_handheld(edge['left'][:2], edge['right'][:2], self.cam, true_pose(edge['times_s']), 800)

    def test_zero_reference_length_is_rejected(self):
        edge = self.room_edge('Width')

        with self.assertRaisesRegex(ValueError, 'Reference length must be a positive number'):
            stereo.measure_handheld(edge['left'], edge['right'], self.cam, true_pose(edge['times_s']), 0)

    def test_point_behind_camera_is_rejected(self):
        edge = self.room_edge('Width')
        truth = true_pose(edge['times_s'])
        reversed_pose = stereo.Pose(truth.rotation, -truth.translation, 100, 5.)

        with self.assertRaisesRegex(ValueError, 'in front of both cameras'):
            self.measure(edge, reversed_pose)

    def test_pose_survives_json_round_trip(self):
        pose = true_pose([.3, 1.7])

        restored = stereo.Pose.from_dict(json.loads(json.dumps(pose.to_dict())))

        np.testing.assert_array_equal(restored.rotation, pose.rotation)


class RoomSummaryTests(unittest.TestCase):
    def test_repeated_measurements_are_averaged_with_spread(self):
        edges = [{'label': 'Width', 'mm': 2990.}, {'label': 'Width', 'mm': 3010.}]

        summary = stereo.room_summary(edges)

        self.assertEqual(summary, [{'label': 'Width', 'mean_mm': 3000., 'count': 2, 'spread_mm': 20.}])

    def test_dimensions_are_listed_width_length_height(self):
        edges = [{'label': 'Height', 'mm': 2400.}, {'label': 'Width', 'mm': 3000.}, {'label': 'Length', 'mm': 4000.}]

        summary = stereo.room_summary(edges)

        self.assertEqual([d['label'] for d in summary], ['Width', 'Length', 'Height'])

    def test_unmeasured_dimensions_are_omitted(self):
        summary = stereo.room_summary([{'label': 'Height', 'mm': 2400.}])

        self.assertEqual([d['label'] for d in summary], ['Height'])

    def test_no_edges_gives_empty_summary(self):
        self.assertEqual(stereo.room_summary([]), [])


class WebTests(SyntheticCase):
    def test_exif_portrait_upload(self):
        import struct
        sys.path.insert(0, str(stereo.ROOT / 'app'))
        from app import create_app
        from modules import module7
        # Store landscape pixels with orientation 6 (90 degrees clockwise).
        fs = cv2.FileStorage(str(self.folder / 'portrait.yaml'), cv2.FILE_STORAGE_WRITE)
        fs.write('camera_matrix', np.array([[800., 0, 360], [0, 800, 640], [0, 0, 1]]))
        fs.write('distortion_coefficients', np.zeros((1, 5)))
        fs.write('image_width', 720)
        fs.write('image_height', 1280)
        fs.release()
        image = np.zeros((720, 1280, 3), np.uint8)
        ok, jpeg = cv2.imencode('.jpg', image)
        self.assertTrue(ok)
        tiff = b'II' + struct.pack('<HIH', 42, 8, 1)
        tiff += struct.pack('<HHI', 274, 3, 1) + struct.pack('<H', 6) + b'\x00\x00' + struct.pack('<I', 0)
        exif = b'Exif\x00\x00' + tiff
        raw = jpeg.tobytes()
        oriented = raw[:2] + b'\xff\xe1' + struct.pack('>H', len(exif) + 2) + exif + raw[2:]
        client = create_app().test_client()
        def upload(content):
            return client.post('/module-7/', data={
                'baseline': '200', 'shape': 'rectangle',
                'camera': (io.BytesIO((self.folder / 'portrait.yaml').read_bytes()), 'camera.yaml'),
                'left': (io.BytesIO(content), 'left.jpg'),
                'right': (io.BytesIO(content), 'right.jpg')})
        response = upload(oriented)
        self.assertEqual(response.status_code, 200, response.text)
        run = re.search(r'name="run" value="([a-f0-9]+)"', response.text).group(1)
        try:
            saved = cv2.imread(str(module7.run_folder(run) / 'left.png'))
            self.assertEqual(saved.shape[:2], (1280, 720))
        finally:
            shutil.rmtree(module7.run_folder(run))
        response = upload(raw)
        self.assertEqual(response.status_code, 400)
        self.assertIn('1280 x 720 pixels', response.text)
        self.assertIn('calibration resolution 720 x 1280', response.text)

    def setUp(self):
        super().setUp()
        app = create_app()
        app.testing = True
        self.client = app.test_client()
        self.runs = []

    def tearDown(self):
        for run in self.runs:
            shutil.rmtree(module7.run_folder(run), ignore_errors=True)
        super().tearDown()

    def upload(self, route, data):
        data = {'baseline': '200', 'camera': (io.BytesIO((self.folder / 'camera.yaml').read_bytes()), 'camera.yaml'), **data}
        response = self.client.post(route, data=data)
        match = re.search(r'name="run" value="([a-f0-9]+)"', response.text)
        if match:
            self.runs.append(match.group(1))
        return response, match.group(1) if match else None

    def upload_video(self):
        return self.upload('/module-7/room', {'action': 'upload', 'video': (io.BytesIO((self.folder / 'room.avi').read_bytes()), 'room.avi')})

    def load_pair(self, run, edge, times=None):
        times = times or edge['times_s']
        return self.client.post('/module-7/room', data={'action': 'pair', 'run': run, 'label': edge['label'],
                                                        'left_time': str(times[0]), 'right_time': str(times[1])})

    def measure_room_edge(self, run, edge):
        self.load_pair(run, edge)
        points = json.dumps({'left': edge['left'], 'right': edge['right']})
        return self.client.post('/module-7/room', data={'action': 'measure', 'run': run, 'points': points,
                                                        'reference_mm': str(edge['reference_mm'])})

    def result_json(self, run):
        with self.client.get(f'/module-7/output/{run}/result.json') as response:
            return response.json

    def test_module_is_listed_on_home_page(self):
        self.assertIn(b'Module 7', self.client.get('/').data)

    def test_photo_upload_then_measure_downloads_rectangle_width(self):
        files = {side: (io.BytesIO((self.folder / f'{side}.png').read_bytes()), f'{side}.png') for side in ('left', 'right')}
        _, run = self.upload('/module-7/', {'shape': 'rectangle', **files})

        self.client.post('/module-7/', data={'action': 'measure', 'run': run, 'points': json.dumps(self.points['object'])})

        self.assertAlmostEqual(self.result_json(run)['dimensions'][0]['mm'], 1000)

    def test_room_edges_from_different_stops_give_room_dimensions(self):
        _, run = self.upload_video()

        for edge in self.points['room']:
            self.measure_room_edge(run, edge)

        summary = {d['label']: d['mean_mm'] for d in self.result_json(run)['summary']}
        np.testing.assert_allclose([summary[k] for k in demo.ROOM_SIZE], list(demo.ROOM_SIZE.values()), rtol=.02)

    def test_room_edge_records_its_stop_times(self):
        _, run = self.upload_video()

        self.measure_room_edge(run, self.room_edge('Length'))

        self.assertEqual(self.result_json(run)['edges'][0]['times_s'], [2.3, 3.7])

    def test_room_pair_serves_the_frames_at_the_chosen_stops(self):
        _, run = self.upload_video()
        expected = [stereo.frame_at(self.folder / 'room.avi', t) for t in (2.5, 3.5)]

        self.load_pair(run, self.room_edge('Length'), (2.5, 3.5))

        served = [cv2.imdecode(np.frombuffer(self.client.get(f'/module-7/output/{run}/{side}.png').data, np.uint8), cv2.IMREAD_COLOR)
                  for side in ('left', 'right')]
        self.assertEqual([np.array_equal(s, e) for s, e in zip(served, expected)], [True, True])

    def test_removing_a_room_edge_drops_it_from_the_summary(self):
        _, run = self.upload_video()
        self.measure_room_edge(run, self.room_edge('Width'))

        self.client.post('/module-7/room', data={'action': 'remove', 'run': run, 'index': '0'})

        self.assertEqual(self.result_json(run)['summary'], [])

    def test_room_measure_without_reference_length_is_a_bad_request(self):
        _, run = self.upload_video()
        edge = self.room_edge('Width')
        self.load_pair(run, edge)

        response = self.client.post('/module-7/room', data={'action': 'measure', 'run': run,
                                                           'points': json.dumps({'left': edge['left'], 'right': edge['right']})})

        self.assertEqual(response.status_code, 400)

    def test_room_page_prefills_reference_with_measured_height(self):
        _, run = self.upload_video()
        self.measure_room_edge(run, self.room_edge('Height'))
        height = f"{self.result_json(run)['summary'][0]['mean_mm']:.0f}"

        response = self.load_pair(run, self.room_edge('Width'))

        self.assertIn(f'name="reference_mm" value="{height}"', response.text)

    def test_room_measure_without_loaded_pair_is_rejected(self):
        _, run = self.upload_video()
        edge = self.room_edge('Width')

        response = self.client.post('/module-7/room', data={'action': 'measure', 'run': run, 'reference_mm': '800',
                                                           'points': json.dumps({'left': edge['left'], 'right': edge['right']})})

        self.assertEqual((response.status_code, 'Load a stereo pair' in response.text), (400, True))

    def test_room_pair_with_equal_stop_times_is_rejected(self):
        _, run = self.upload_video()

        response = self.load_pair(run, self.room_edge('Width'), (1, 1))

        self.assertEqual((response.status_code, 'two different frame times' in response.text), (400, True))

    def test_room_error_keeps_already_measured_edges_on_page(self):
        _, run = self.upload_video()
        self.measure_room_edge(run, self.room_edge('Width'))
        measured = f"{self.result_json(run)['edges'][0]['mm']:.1f}"

        response = self.load_pair(run, self.room_edge('Width'), (1, 1))

        self.assertIn(measured, response.text)

    def test_room_video_must_match_calibration_resolution(self):
        fs = cv2.FileStorage(str(self.folder / 'camera.yaml'), cv2.FILE_STORAGE_WRITE)
        fs.write('camera_matrix', demo.K)
        fs.write('distortion_coefficients', np.zeros((1, 5)))
        fs.write('image_width', 640)
        fs.write('image_height', 360)
        fs.release()

        response, run = self.upload_video()

        self.assertEqual((response.status_code, run, 'must match calibration resolution' in response.text), (400, None, True))

    def test_calibration_file_is_not_downloadable(self):
        _, run = self.upload_video()

        self.assertEqual(self.client.get(f'/module-7/output/{run}/camera.yaml').status_code, 404)

    def test_empty_photo_post_is_a_bad_request(self):
        self.assertEqual(self.client.post('/module-7/', data={}).status_code, 400)

    def test_invalid_run_output_is_not_found(self):
        self.assertEqual(self.client.get('/module-7/output/invalid/left.png').status_code, 404)


if __name__ == '__main__':
    unittest.main()
