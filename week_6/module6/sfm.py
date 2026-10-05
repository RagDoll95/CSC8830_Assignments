"""Run: python week_6/module6/sfm.py --out output
Four-view planar orthographic factorization (FPCV-4-4, centering and SVD).
Default: SIMULATED flat L-shaped card, known scale and front-parallel first view.
Optional --input data.json: {"points": four lists of corresponding [x,y] pixels,
"scale": pixels_per_unit}. Supply boundary points in order, same order in every view.
Requires the root requirements.txt. This is a rank-2 planar example, not general 3D SfM.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def reconstruct(points, scale):
    points = np.asarray(points, dtype=float)
    if points.ndim != 3 or points.shape[0] != 4 or points.shape[2] != 2 or points.shape[1] < 3:
        raise ValueError('Provide four views of at least three corresponding boundary points.')
    if not np.isfinite(points).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Coordinates must be finite and scale must be positive.')
    centers = points.mean(axis=1)
    w = ((points-centers[:, None, :])/scale).transpose(0, 2, 1).reshape(8, -1)
    u, singular, vt = np.linalg.svd(w, full_matrices=False)
    if singular[1] < 1e-8:
        raise ValueError('Boundary points must not all lie on a line.')
    motion = u[:, :2] * np.sqrt(singular[:2])
    shape = np.sqrt(singular[:2, None]) * vt[:2]
    # Fix the affine ambiguity using the KNOWN front-parallel first camera.
    q = np.linalg.inv(motion[:2])
    motion = motion @ q
    shape = np.linalg.solve(q, shape)
    predicted = (scale*(motion@shape)).reshape(4, 2, -1).transpose(0, 2, 1)+centers[:, None, :]
    return shape, {'centers_px': centers.tolist(), 'W': w.tolist(),
                   'singular_values': singular.tolist(), 'Q': q.tolist(),
                   'motion': motion.tolist(), 'shape': shape.tolist(),
                   'reprojection_rms_px': float(np.sqrt(np.mean(np.sum((points-predicted)**2, axis=2))))}


def run_example(output, input_file=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if input_file:
        data = json.loads(Path(input_file).read_text())
    else:
        shape = np.array([[-2,-1], [2,-1], [2,0], [0,0], [0,1], [-2,1]], float)
        points, cameras = [], []
        for index, (yaw, pitch) in enumerate([(0,0), (30,0), (-30,0), (0,30)]):
            a, b = np.radians([yaw, pitch])
            ry = np.array([[np.cos(a),0,np.sin(a)], [0,1,0], [-np.sin(a),0,np.cos(a)]])
            rx = np.array([[1,0,0], [0,np.cos(b),-np.sin(b)], [0,np.sin(b),np.cos(b)]])
            rotation = rx @ ry
            center = 10*rotation[2]
            pixels = np.rint(70*shape@rotation[:2, :2].T+[320,180]).astype(int)
            points.append(pixels.tolist())
            cameras.append({'view': index+1, 'yaw_deg': yaw, 'pitch_deg': pitch,
                            'center_cm': center.tolist(), 'world_to_camera_rotation': rotation.tolist()})
            image = np.full((360,640,3), 245, np.uint8)
            cv2.fillPoly(image, [pixels], (215,225,235))
            cv2.polylines(image, [pixels], True, (70,70,70), 2)
            for j, p in enumerate(pixels):
                cv2.circle(image, tuple(p), 4, (0,0,220), -1)
                cv2.putText(image, str(j+1), tuple(p+[7,-7]), cv2.FONT_HERSHEY_SIMPLEX, .5, (20,20,20), 1)
            cv2.putText(image, f'Simulated view {index+1}', (15,25), cv2.FONT_HERSHEY_SIMPLEX, .6, (20,20,20), 1)
            cv2.imwrite(str(output / f'view{index+1}.png'), image)
        data = {'simulated': True, 'points': points, 'scale': 70, 'units': 'cm',
                'image_size': [640,360], 'principal_point': [320,180],
                'projection': 'orthographic; no perspective focal length or distortion',
                'cameras': cameras, 'reference_shape': shape.tolist()}
    recovered, calculations = reconstruct(data['points'], data['scale'])
    (output / 'views.json').write_text(json.dumps(data, indent=2))
    (output / 'sfm_calculations.json').write_text(json.dumps(calculations, indent=2))
    np.savetxt(output / 'boundary.csv', np.column_stack([recovered.T, np.zeros(recovered.shape[1])]),
               delimiter=',', header='x,y,z', comments='')
    fig, ax = plt.subplots(figsize=(6,4), layout='constrained')
    closed = np.column_stack([recovered, recovered[:, 0]])
    ax.plot(*closed, 'o-')
    for j, p in enumerate(recovered.T): ax.annotate(str(j+1), p, xytext=(6,6), textcoords='offset points')
    ax.set_aspect('equal'); ax.grid(True)
    ax.set_xlabel('x (object units)'); ax.set_ylabel('y (object units)')
    ax.set_title('Planar boundary (z = 0)')
    fig.savefig(output / 'reconstruction.png', dpi=160)
    plt.close(fig)
    return calculations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--input')
    args = parser.parse_args()
    print(json.dumps(run_example(args.out, args.input), indent=2))
