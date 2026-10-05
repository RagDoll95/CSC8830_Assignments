"""Run: python week_6/module6/motion.py video.mp4 --out output --start 0
Requires: pip install -r requirements.txt. Processes 30 seconds of sparse optical flow.
"""
import argparse
import csv
from pathlib import Path
import shutil
import subprocess
import cv2
import numpy as np


def process_video(video, output, start=0):
    if not np.isfinite(start) or start < 0:
        raise ValueError('Start time must be nonnegative.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    writer = None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError('Could not read this video.')
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(start * fps))
        ok, previous = capture.read()
        if not ok:
            raise ValueError('Start time is beyond the video.')
        h, w = previous.shape[:2]
        scale = min(1, 640 / w)
        size = (max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2))
        previous = cv2.resize(previous, size)
        gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        writer = cv2.VideoWriter(str(output / 'flow.avi'),
                                cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
        if not writer.isOpened():
            raise ValueError('Could not create the output video.')
        with (output / 'points.csv').open('w', newline='') as stream:
            table = csv.writer(stream)
            table.writerow(['frame', 'x0', 'y0', 'x1', 'y1', 'dx', 'dy'])
            for frame_index in range(int(np.ceil(30 * fps)) - 1):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError('Choose a start time with at least 30 seconds remaining.')
                frame = cv2.resize(frame, size)
                current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                points = cv2.goodFeaturesToTrack(gray, 100, .02, 10)
                overlay = previous.copy()
                if points is not None:
                    following, status, _ = cv2.calcOpticalFlowPyrLK(
                        gray, current, points, None, winSize=(21, 21), maxLevel=3)
                    if following is not None:
                        for first, second, good in zip(points[:, 0], following[:, 0], status[:, 0]):
                            if not good or not np.all(np.isfinite(second)):
                                continue
                            if not (0 <= second[0] < size[0] and 0 <= second[1] < size[1]):
                                continue
                            table.writerow([frame_index, *first, *second, *(second - first)])
                            cv2.arrowedLine(overlay, tuple(np.rint(first).astype(int)),
                                            tuple(np.rint(second).astype(int)), (0, 255, 255), 1)
                writer.write(overlay)
                if frame_index == 0:
                    cv2.imwrite(str(output / 'frame0.png'), previous)
                    cv2.imwrite(str(output / 'frame1.png'), frame)
                    cv2.imwrite(str(output / 'pair_flow.png'), overlay)
                previous, gray = frame, current
            writer.write(previous)
        writer.release()
        writer = None
        # Convert for browser playback when ffmpeg is installed; AVI still works locally.
        result = 'flow.avi'
        if shutil.which('ffmpeg'):
            encoded = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(output / result),
                                      '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                                      str(output / 'flow.mp4')], capture_output=True)
            if encoded.returncode == 0:
                result = 'flow.mp4'
        return result
    finally:
        capture.release()
        if writer is not None:
            writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video')
    parser.add_argument('--out', required=True)
    parser.add_argument('--start', type=float, default=0)
    args = parser.parse_args()
    print(process_video(args.video, args.out, args.start))
