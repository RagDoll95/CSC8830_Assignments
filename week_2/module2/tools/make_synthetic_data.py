"""
Tooling -- generate a SYNTHETIC dataset so the pipeline can be verified end to
end without a phone.

    #####################################################################
    #  THIS IS NOT SUBMISSION DATA.                                     #
    #  Every file it writes is named synthetic_*. It exists to prove the #
    #  code is correct (recovered K vs. known ground-truth K), not to    #
    #  stand in for the real captures the assignment requires.          #
    #  Replace with your own photos before submitting -- see README.     #
    #####################################################################

WHY IT IS USEFUL
    The ground-truth K and distortion coefficients are known exactly, so
    calibrate.py can be scored rather than merely run: if the recovered f_x is
    within a fraction of a percent of the truth, the calibration code is right
    and any later error is a capture problem, not a code problem.

HOW TO RUN
    # from module2/
    python tools/make_synthetic_data.py                  # everything
    python tools/make_synthetic_data.py --only board      # calibration images
    python tools/make_synthetic_data.py --only measure    # Step 2/3 scenes

WHAT IT WRITES
    calibration/data/images/synthetic_board_##.jpg   18 checkerboard poses
    calibration/data/synthetic_ground_truth.yaml     the true K and dist
    measurement/data/synthetic_selftest_board.jpg    board at a known Z
    validation/data/images/synthetic_meas_##.jpg     20 measurement scenes
    validation/data/synthetic_truth.csv              true sizes, true Z, true pixels

HOW THE IMAGES ARE MADE
    1. A checkerboard/poster texture is rendered in millimetre space.
    2. For each pose (R, t) the board plane Z=0 maps to the image by the
       homography K [r1 r2 t], applied with warpPerspective -- exact pinhole
       projection.
    3. Lens distortion is then applied to the whole frame by remapping: for
       every output (distorted) pixel, cv2.undistortPoints gives the matching
       ideal-image pixel. One map, reused for every frame.
    4. Mild blur, vignetting and sensor noise, then JPEG -- so corner
       detection and cornerSubPix face a realistic, not a perfect, image.
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Ground truth -- a plausible smartphone rear camera.
# --------------------------------------------------------------------------
IMAGE_SIZE = (2016, 1512)          # (w, h)
FX_TRUE, FY_TRUE = 1600.5, 1601.2  # near-identical: square pixels
CX_TRUE, CY_TRUE = 1007.3, 755.8   # slightly off centre, as real sensors are
DIST_TRUE = np.array([[0.080, -0.160, 0.00080, -0.00050, 0.050]])  # mild barrel

K_TRUE = np.array([[FX_TRUE, 0.0, CX_TRUE],
                   [0.0, FY_TRUE, CY_TRUE],
                   [0.0, 0.0, 1.0]])

PATTERN_SIZE = (9, 6)    # inner corners (cols, rows)
SQUARE_MM = 25.0
MARGIN_MM = 20.0         # white border around the printed grid
PX_PER_MM = 4.0          # texture resolution

RNG = np.random.default_rng(20260913)

# Capture-realism knobs used by the Step 3 scenes.
SIGMA_CLICK_PX = 1.5     # human point-picking precision
Z_BIAS_MM = -8.0         # tape starts at the back glass, not the optical centre
SIGMA_Z_MM = 3.0         # random tape-measure error


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------
def board_texture():
    """Render the printed checkerboard in millimetre space.

    Returns (texture_bgr, world_from_tex) where world_from_tex is the 3x3 that
    takes a texture pixel to a board-plane (X, Y) in mm, with inner corner
    (0,0) at the origin -- matching calibrate.py's object points.
    """
    cols, rows = PATTERN_SIZE
    sq_x, sq_y = cols + 1, rows + 1          # 9x6 inner corners => 10x7 squares
    board_w_mm = sq_x * SQUARE_MM + 2 * MARGIN_MM
    board_h_mm = sq_y * SQUARE_MM + 2 * MARGIN_MM
    w = int(round(board_w_mm * PX_PER_MM))
    h = int(round(board_h_mm * PX_PER_MM))

    tex = np.full((h, w, 3), 245, np.uint8)  # paper white
    s = SQUARE_MM * PX_PER_MM
    off = MARGIN_MM * PX_PER_MM
    for r in range(sq_y):
        for c in range(sq_x):
            if (r + c) % 2 == 0:
                continue
            x0 = int(round(off + c * s))
            y0 = int(round(off + r * s))
            x1 = int(round(off + (c + 1) * s))
            y1 = int(round(off + (r + 1) * s))
            tex[y0:y1, x0:x1] = 18        # ink black

    # Texture pixel -> board mm, then shift so inner corner (0,0) is the origin.
    shift = MARGIN_MM + SQUARE_MM
    world_from_tex = np.array([[1.0 / PX_PER_MM, 0.0, -shift],
                               [0.0, 1.0 / PX_PER_MM, -shift],
                               [0.0, 0.0, 1.0]])
    return tex, world_from_tex


def poster_texture(width_mm, height_mm, label, seed):
    """Render a flat rectangular target of exact known size, on white.

    The measured object is the coloured rectangle; its extent in mm is exactly
    (width_mm, height_mm), so ground truth is known by construction.
    """
    rng = np.random.default_rng(seed)
    pad_mm = 30.0
    ppmm = 2.0
    w = int(round((width_mm + 2 * pad_mm) * ppmm))
    h = int(round((height_mm + 2 * pad_mm) * ppmm))
    tex = np.full((h, w, 3), 232, np.uint8)

    x0, y0 = int(round(pad_mm * ppmm)), int(round(pad_mm * ppmm))
    x1 = int(round((pad_mm + width_mm) * ppmm))
    y1 = int(round((pad_mm + height_mm) * ppmm))
    colour = tuple(int(c) for c in rng.integers(40, 200, size=3))
    cv2.rectangle(tex, (x0, y0), (x1, y1), colour, -1)
    cv2.rectangle(tex, (x0, y0), (x1, y1), (20, 20, 20), max(2, int(ppmm)))

    # Corner ticks, so the clicked points are visually unambiguous on the demo.
    tick = int(round(12 * ppmm))
    for (cx, cy) in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        cv2.line(tex, (cx - tick, cy), (cx + tick, cy), (255, 255, 255), 2)
        cv2.line(tex, (cx, cy - tick), (cx, cy + tick), (255, 255, 255), 2)
    cv2.putText(tex, label, (x0 + 10, y0 + int(40 * ppmm)),
                cv2.FONT_HERSHEY_SIMPLEX, ppmm * 0.5, (255, 255, 255),
                max(2, int(ppmm)), cv2.LINE_AA)

    world_from_tex = np.array([[1.0 / ppmm, 0.0, -pad_mm],
                               [0.0, 1.0 / ppmm, -pad_mm],
                               [0.0, 0.0, 1.0]])
    # Object corners in plane coords (mm), origin at the rectangle's top-left.
    corners_mm = np.array([[0.0, 0.0],
                           [width_mm, 0.0],
                           [width_mm, height_mm],
                           [0.0, height_mm]])
    return tex, world_from_tex, corners_mm


def rodrigues(rx_deg, ry_deg, rz_deg):
    rvec = np.deg2rad(np.array([rx_deg, ry_deg, rz_deg], dtype=np.float64))
    R, _ = cv2.Rodrigues(rvec)
    return R


def build_distortion_map(K, dist, image_size):
    """Map an ideal pinhole frame to a lens-distorted frame.

    For every output (distorted) pixel, undistortPoints returns the normalized
    ideal ray; projecting that with K gives the ideal-image pixel to sample.
    Computed once and reused for every frame.
    """
    w, h = image_size
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    pts = np.stack([uu.ravel(), vv.ravel()], axis=1).reshape(-1, 1, 2)
    norm = cv2.undistortPoints(pts.astype(np.float64), K, dist).reshape(-1, 2)
    map_x = (norm[:, 0] * K[0, 0] + K[0, 2]).reshape(h, w).astype(np.float32)
    map_y = (norm[:, 1] * K[1, 1] + K[1, 2]).reshape(h, w).astype(np.float32)
    return map_x, map_y


def render_plane(tex, world_from_tex, R, t, K, image_size, bg=120):
    """Warp a planar texture into an ideal pinhole frame at pose (R, t)."""
    w, h = image_size
    # Plane Z=0: [X Y 1] -> K [r1 r2 t] [X Y 1]
    P = K @ np.column_stack([R[:, 0], R[:, 1], t])
    H = P @ world_from_tex

    canvas = np.full((h, w, 3), bg, np.uint8)
    warped = cv2.warpPerspective(tex, H, (w, h), flags=cv2.INTER_AREA,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.full(tex.shape[:2], 255, np.uint8), H, (w, h),
                               flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    canvas[mask > 127] = warped[mask > 127]
    return canvas, H


def add_realism(img, map_x, map_y, blur=1.0, noise=2.0, vignette=0.25, seed=0):
    """Distort, blur, vignette and add sensor noise. Keeps the detector honest."""
    rng = np.random.default_rng(seed)
    out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(120, 120, 120))
    if blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    if vignette > 0:
        h, w = out.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        gain = (1.0 - vignette * np.clip(r, 0, 1.4) ** 2)[..., None]
        out = np.clip(out.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    if noise > 0:
        out = np.clip(out.astype(np.float32)
                      + rng.normal(0, noise, out.shape), 0, 255).astype(np.uint8)
    return out


# --------------------------------------------------------------------------
# Calibration image set
# --------------------------------------------------------------------------
# Tilt +/-30-45 deg in both axes, varied distance, and the board deliberately
# placed in every region of the frame -- corners especially, because that is
# where distortion lives. Only one near-fronto-parallel shot: a board that is
# parallel to the sensor in every image is a degenerate configuration.
#
# (rx, ry, roll, Z_mm, target frame cell as (row, col) in a 3x3 grid)
BOARD_POSES = [
    (-26,  17,   4,  385, (0, 0)),
    ( 31, -26,  -6,  330, (0, 1)),
    (-40, -31,   9,  310, (0, 2)),
    ( 36,  33, -11,  360, (1, 0)),
    (-45,   7,   2,  270, (1, 1)),
    ( 42,  -9,   5,  285, (1, 2)),
    (-13,  43,  -3,  390, (2, 0)),
    ( 15, -41,   7,  410, (2, 1)),
    (-27,  28,  15,  430, (2, 2)),
    ( 22, -24, -14,  320, (0, 1)),
    (-37, -15,  -8,  460, (1, 0)),
    ( 34,  18,  10,  490, (2, 2)),
    (-21,  35,   6,  340, (0, 2)),
    ( 18, -28,  -5,  410, (2, 0)),
    (-43,  23,  12,  425, (1, 2)),
    ( 41, -21, -13,  305, (2, 1)),
    ( -9,  -8,   3,  520, (1, 1)),   # the one near-fronto-parallel shot
    ( 32,  37,  -9,  400, (0, 0)),
]

# All 54 inner corners must project inside the frame or findChessboardCorners
# fails outright, so the requested cell centre is pulled back toward the frame
# centre until every corner clears this margin.
CORNER_MARGIN_PX = 45


def _grid_centre_offset():
    """Board-frame offset that puts the inner-corner grid's centre at the pose."""
    cols, rows = PATTERN_SIZE
    return ((cols - 1) * SQUARE_MM / 2.0, (rows - 1) * SQUARE_MM / 2.0)


def solve_pose(rx, ry, roll, Z_mm, cell, objp):
    """Find (R, t) putting the board's grid centre near a target frame cell.

    Returns (R, t) with every inner corner at least CORNER_MARGIN_PX inside
    the frame; if the requested cell cannot be reached the target is drawn
    back toward the frame centre rather than emitting an undetectable board.
    """
    w, h = IMAGE_SIZE
    R = rodrigues(rx, ry, roll)
    gx, gy = _grid_centre_offset()
    row, col = cell

    # Cell centre in pixels, inset from the frame edge.
    u_target = (col + 0.5) / 3.0 * w
    v_target = (row + 0.5) / 3.0 * h
    u_centre, v_centre = w / 2.0, h / 2.0

    for pull in np.linspace(0.0, 1.0, 21):
        u_aim = u_target + (u_centre - u_target) * pull
        v_aim = v_target + (v_centre - v_target) * pull
        # Place the grid centre on the ray through (u_aim, v_aim) at depth Z.
        Xc = (u_aim - CX_TRUE) * Z_mm / FX_TRUE
        Yc = (v_aim - CY_TRUE) * Z_mm / FY_TRUE
        # t positions the board origin (inner corner 0,0), so undo the rotated
        # offset from the board origin to the grid centre.
        centre_in_cam = R @ np.array([gx, gy, 0.0])
        t = (np.array([Xc, Yc, Z_mm]) - centre_in_cam).reshape(3, 1)

        proj, _ = cv2.projectPoints(objp, cv2.Rodrigues(R)[0], t,
                                    K_TRUE, DIST_TRUE)
        proj = proj.reshape(-1, 2)
        inside = (proj[:, 0] > CORNER_MARGIN_PX) & (proj[:, 0] < w - CORNER_MARGIN_PX) \
            & (proj[:, 1] > CORNER_MARGIN_PX) & (proj[:, 1] < h - CORNER_MARGIN_PX)
        if inside.all():
            return R, t, proj
    return R, t, proj


def make_board_images(out_dir, map_x, map_y, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    tex, world_from_tex = board_texture()
    cols, rows = PATTERN_SIZE
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= SQUARE_MM

    written = []
    for i, (rx, ry, roll, Z_mm, cell) in enumerate(BOARD_POSES, start=1):
        R, t, proj = solve_pose(rx, ry, roll, Z_mm, cell, objp)
        ideal, _ = render_plane(tex, world_from_tex, R, t, K_TRUE, IMAGE_SIZE)
        img = add_realism(ideal, map_x, map_y, blur=0.9, noise=2.0, seed=100 + i)
        path = os.path.join(out_dir, "synthetic_board_%02d.jpg" % i)
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        written.append(path)
        if verbose:
            frac = (np.ptp(proj[:, 0]) / IMAGE_SIZE[0]) * (np.ptp(proj[:, 1]) / IMAGE_SIZE[1])
            print("  wrote %s  tilt %+3d/%+3d deg  Z=%4d mm  cell%s  fills %.0f%% of frame"
                  % (os.path.basename(path), rx, ry, Z_mm, cell, frac * 100))
    return written


def make_selftest_board(out_path, map_x, map_y, Z_mm=1000.0):
    """A fronto-parallel board at a known Z -- input to selftest_checkerboard.py."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tex, world_from_tex = board_texture()
    R = rodrigues(0, 0, 0)
    cols, rows = PATTERN_SIZE
    # Centre the grid in frame: shift by half the inner-corner extent.
    tx = -(cols - 1) * SQUARE_MM / 2.0
    ty = -(rows - 1) * SQUARE_MM / 2.0
    t = np.array([[tx], [ty], [Z_mm]], dtype=np.float64)
    ideal, _ = render_plane(tex, world_from_tex, R, t, K_TRUE, IMAGE_SIZE)
    img = add_realism(ideal, map_x, map_y, blur=0.8, noise=1.5, seed=7)
    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("  wrote %s  (fronto-parallel, true Z = %.0f mm)"
          % (os.path.basename(out_path), Z_mm))
    return out_path, Z_mm


# --------------------------------------------------------------------------
# Step 3 measurement scenes -- 20 measurements, varied object and dimension
# --------------------------------------------------------------------------
# (object, dimension, width_mm, height_mm, true Z in mm)
# Every Z is > 2000 mm as the assignment requires. Objects and the dimension
# under test both vary -- 20 clicks on one book is not 20 measurements.
MEASUREMENT_SCENES = [
    ("hardback book",        "H", 155.0, 240.0, 2150.0),
    ("hardback book",        "W", 155.0, 240.0, 2480.0),
    ("laptop lid",           "W", 358.0, 247.0, 2320.0),
    ("laptop lid",           "D", 358.0, 247.0, 2910.0),
    ("A4 sheet",             "H", 210.0, 297.0, 2240.0),
    ("A4 sheet",             "W", 210.0, 297.0, 3050.0),
    ("monitor bezel",        "W", 597.0, 336.0, 3380.0),
    ("monitor bezel",        "H", 597.0, 336.0, 2760.0),
    ("cereal box front",     "H", 195.0, 300.0, 2080.0),
    ("cereal box front",     "W", 195.0, 300.0, 2640.0),
    ("whiteboard",           "W", 900.0, 600.0, 4120.0),
    ("whiteboard",           "D", 900.0, 600.0, 3540.0),
    ("door panel",           "H", 762.0, 1981.0, 4480.0),
    ("door panel",           "W", 762.0, 1981.0, 3960.0),
    ("poster",               "D", 420.0, 594.0, 2870.0),
    ("poster",               "H", 420.0, 594.0, 2210.0),
    ("keyboard",             "W", 440.0, 138.0, 2130.0),
    ("pizza box",            "W", 330.0, 330.0, 2560.0),
    ("backpack panel",       "H", 300.0, 460.0, 2410.0),
    ("framed picture",       "D", 500.0, 400.0, 3120.0),
]


def dimension_endpoints(corners_mm, dimension):
    """Which two object corners a W / H / D measurement uses.

    corners_mm order is TL, TR, BR, BL.
    """
    if dimension == "W":
        return corners_mm[0], corners_mm[1]      # top edge
    if dimension == "H":
        return corners_mm[0], corners_mm[3]      # left edge
    if dimension == "D":
        return corners_mm[0], corners_mm[2]      # TL -> BR diagonal
    raise ValueError("dimension must be W, H or D")


def make_measurement_scenes(image_dir, truth_csv, map_x, map_y):
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(truth_csv)), exist_ok=True)

    rows = []
    for i, (obj, dim, w_mm, h_mm, Z_true) in enumerate(MEASUREMENT_SCENES, start=1):
        tex, world_from_tex, corners_mm = poster_texture(w_mm, h_mm, obj, seed=i)

        # Fronto-parallel and centred in frame -- the assumption the method
        # rests on, and the only way the tape-measured Z really is Z.
        R = rodrigues(0, 0, 0)
        t = np.array([[-w_mm / 2.0], [-h_mm / 2.0], [Z_true]], dtype=np.float64)
        ideal, _ = render_plane(tex, world_from_tex, R, t, K_TRUE, IMAGE_SIZE)
        img = add_realism(ideal, map_x, map_y, blur=0.8, noise=2.0, seed=500 + i)

        name = "synthetic_meas_%02d.jpg" % i
        cv2.imwrite(os.path.join(image_dir, name), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

        # True pixel locations of the two endpoints, WITH distortion -- these
        # are the pixels a perfect click would land on in the saved image.
        a_mm, b_mm = dimension_endpoints(corners_mm, dim)
        pts3d = np.array([[a_mm[0], a_mm[1], 0.0],
                          [b_mm[0], b_mm[1], 0.0]], dtype=np.float64)
        proj, _ = cv2.projectPoints(pts3d, cv2.Rodrigues(R)[0], t,
                                    K_TRUE, DIST_TRUE)
        proj = proj.reshape(-1, 2)

        gt_mm = float(np.hypot(b_mm[0] - a_mm[0], b_mm[1] - a_mm[1]))

        rows.append({
            "id": i,
            "image_file": name,
            "object": obj,
            "dimension": dim,
            "Z_true_mm": "%.1f" % Z_true,
            "ground_truth_mm": "%.2f" % gt_mm,
            "u1_true": "%.3f" % proj[0, 0], "v1_true": "%.3f" % proj[0, 1],
            "u2_true": "%.3f" % proj[1, 0], "v2_true": "%.3f" % proj[1, 1],
        })
        print("  wrote %s  %-18s %s  truth %7.1f mm  Z %6.0f mm"
              % (name, obj, dim, gt_mm, Z_true))

    with open(truth_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("  wrote %s" % truth_csv)
    return rows


def save_ground_truth(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", K_TRUE)
    fs.write("distortion_coefficients", DIST_TRUE)
    fs.write("image_width", IMAGE_SIZE[0])
    fs.write("image_height", IMAGE_SIZE[1])
    fs.write("square_size_mm", SQUARE_MM)
    fs.write("pattern_cols", PATTERN_SIZE[0])
    fs.write("pattern_rows", PATTERN_SIZE[1])
    fs.write("rms_reprojection_error_px", 0.0)
    fs.write("image_count", len(BOARD_POSES))
    fs.write("notes", "SYNTHETIC ground truth -- not a calibration result")
    fs.release()
    print("  wrote %s" % path)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate the synthetic dataset")
    ap.add_argument("--only", choices=["board", "measure", "all"], default="all")
    ap.add_argument("--board-dir", default="calibration/data/images")
    ap.add_argument("--truth-yaml", default="calibration/data/synthetic_ground_truth.yaml")
    ap.add_argument("--selftest-image", default="measurement/data/synthetic_selftest_board.jpg")
    ap.add_argument("--meas-dir", default="validation/data/images")
    ap.add_argument("--truth-csv", default="validation/data/synthetic_truth.csv")
    args = ap.parse_args(argv)

    print("=" * 66)
    print("SYNTHETIC DATASET GENERATOR -- for pipeline verification only")
    print("Not submission data. Replace with real captures before submitting.")
    print("=" * 66)
    print("ground-truth camera: %dx%d  fx=%.1f fy=%.1f cx=%.1f cy=%.1f"
          % (IMAGE_SIZE[0], IMAGE_SIZE[1], FX_TRUE, FY_TRUE, CX_TRUE, CY_TRUE))
    print("ground-truth dist  : %s\n" % np.array2string(DIST_TRUE.ravel(), precision=5))

    print("building the distortion remap (once)...")
    map_x, map_y = build_distortion_map(K_TRUE, DIST_TRUE, IMAGE_SIZE)

    if args.only in ("board", "all"):
        print("\ncalibration images (%d poses):" % len(BOARD_POSES))
        make_board_images(args.board_dir, map_x, map_y)
        save_ground_truth(args.truth_yaml)
        print("\nStep 2 self-test image:")
        make_selftest_board(args.selftest_image, map_x, map_y, Z_mm=1000.0)

    if args.only in ("measure", "all"):
        print("\nStep 3 measurement scenes (%d):" % len(MEASUREMENT_SCENES))
        make_measurement_scenes(args.meas_dir, args.truth_csv, map_x, map_y)

    print("\ndone. next:")
    print("  python calibration/capture_check.py")
    print("  python calibration/calibrate.py")
    print("  python tools/verify_pipeline.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
