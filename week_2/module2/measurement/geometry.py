"""
Step 2 -- real-world 2D measurement from perspective projection. PURE GEOMETRY.

WHAT IT DOES
    Loads a camera_params.yaml from Step 1 and converts image pixels into
    real-world millimetres at a known object depth Z.

    This module has NO UI: no cv2.imshow, no Flask, no argparse side effects.
    Both measure_cli.py and app/app.py import these same functions, so the
    numbers cannot drift between the CLI and the web app.

HOW TO RUN
    Not a script. Import it:

        from geometry import load_params, backproject, measure
        cam = load_params("calibration/output/camera_params.yaml")
        r = measure((100, 200), (800, 210), Z_mm=2400.0, cam=cam)
        print(r["length_mm"])

    Self-test (no photos required):
        python measurement/geometry.py

THE MATH (plan sections 6 and 12)
    Forward pinhole projection:
        u = f_x * X/Z + c_x        v = f_y * Y/Z + c_y
    Inverted at known depth Z -- the one primitive everything else reuses:
        X = (u - c_x) * Z / f_x    Y = (v - c_y) * Z / f_y    Z = Z
    A dimension is then the Euclidean distance between two backprojections, so
    width, height and diagonal come from one code path rather than three.
    For an axis-aligned span this collapses to the form quoted in the report:
        W = du * Z / f_x           H = dv * Z / f_y

DISTORTION -- OPTION 1, chosen once and applied in exactly one place
    We undistort THE TWO SELECTED POINTS ONLY, via cv2.undistortPoints, which
    returns *normalized* coordinates (already X/Z and Y/Z). The dimension is
    therefore Z * d(normalized), with NO further division by f_x -- dividing
    again is the classic double-correction bug.
    We never undistort the image as well; the user clicks the real, distorted
    image, so correcting the two picked points is both simpler and more
    accurate than resampling the whole frame.

ERROR PROPAGATION (for the report)
    Since W = du * Z / f_x, relative errors add:
        dW/W  ~=  d(du)/du  +  dZ/Z  +  df_x/f_x
    This is why larger objects measure better: a bigger du shrinks the
    click-precision term.
"""

import math
import os

import cv2
import numpy as np


class CalibrationError(Exception):
    """Raised for a missing/corrupt params file or a resolution mismatch."""


class Camera(object):
    """Immutable-ish bundle of everything Step 1 produced."""

    def __init__(self, K, dist, image_size, square_size_mm=None,
                 pattern_size=None, rms=None, image_count=None, notes="",
                 source_path=None, scaled_from=None):
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.asarray(dist, dtype=np.float64).reshape(1, -1)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.square_size_mm = square_size_mm
        self.pattern_size = pattern_size
        self.rms = rms
        self.image_count = image_count
        self.notes = notes
        self.source_path = source_path
        # Set when the intrinsics were rescaled to a different resolution.
        self.scaled_from = scaled_from

    # -- convenience accessors -------------------------------------------
    @property
    def fx(self):
        return float(self.K[0, 0])

    @property
    def fy(self):
        return float(self.K[1, 1])

    @property
    def cx(self):
        return float(self.K[0, 2])

    @property
    def cy(self):
        return float(self.K[1, 2])

    @property
    def width(self):
        return self.image_size[0]

    @property
    def height(self):
        return self.image_size[1]

    def summary(self):
        return {
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "width": self.width, "height": self.height,
            "dist": self.dist.ravel().tolist(),
            "rms_px": self.rms, "image_count": self.image_count,
            "square_size_mm": self.square_size_mm,
            "pattern_size": self.pattern_size,
            "notes": self.notes, "scaled_from": self.scaled_from,
        }

    def scaled_to(self, image_size):
        """Return a copy of these intrinsics rescaled to another resolution.

        K is resolution-specific: f and c are in pixels. Measuring a
        1920x1080 frame with intrinsics fitted at 4032x3024 is wrong by the
        scale factor -- the single most likely silent failure in the whole
        assignment. Rescaling is a fallback, not a substitute for shooting
        Steps 2-3 at the calibration resolution.
        """
        w_new, h_new = int(image_size[0]), int(image_size[1])
        sx = w_new / float(self.width)
        sy = h_new / float(self.height)
        K = self.K.copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        return Camera(K, self.dist, (w_new, h_new), self.square_size_mm,
                      self.pattern_size, self.rms, self.image_count,
                      self.notes, self.source_path,
                      scaled_from=(self.width, self.height))


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_params(path):
    """Read a camera_params.yaml written by calibration/calibrate.py.

    Returns a Camera. Raises CalibrationError on anything unusable.
    """
    if not os.path.exists(path):
        raise CalibrationError(
            "camera parameters not found: %s\n"
            "Run Step 1 first:  python calibration/calibrate.py" % path)

    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise CalibrationError("could not open %s as an OpenCV YAML file" % path)
    try:
        K = fs.getNode("camera_matrix").mat()
        dist = fs.getNode("distortion_coefficients").mat()
        if K is None or dist is None:
            raise CalibrationError(
                "%s is missing camera_matrix / distortion_coefficients" % path)

        def num(key, default=None):
            node = fs.getNode(key)
            return default if node.empty() else float(node.real())

        def text(key, default=""):
            node = fs.getNode(key)
            return default if node.empty() else node.string()

        w = num("image_width")
        h = num("image_height")
        if not w or not h:
            raise CalibrationError("%s is missing image_width / image_height" % path)

        cols = num("pattern_cols")
        rows = num("pattern_rows")
        pattern = (int(cols), int(rows)) if cols and rows else None

        return Camera(
            K=K, dist=dist, image_size=(int(w), int(h)),
            square_size_mm=num("square_size_mm"),
            pattern_size=pattern,
            rms=num("rms_reprojection_error_px"),
            image_count=int(num("image_count", 0)) or None,
            notes=text("notes"),
            source_path=path,
        )
    finally:
        fs.release()


# --------------------------------------------------------------------------
# Guards (plan section 9)
# --------------------------------------------------------------------------
def check_resolution(cam, image_size, allow_scaling=False):
    """Assert the measurement image matches the calibration resolution.

    Returns (camera_to_use, warning_or_None). With allow_scaling=False a
    mismatch is refused outright; with True the intrinsics are rescaled and a
    loud warning is returned for the caller to surface.
    """
    w, h = int(image_size[0]), int(image_size[1])
    if (w, h) == cam.image_size:
        return cam, None

    msg = ("RESOLUTION MISMATCH: image is %dx%d but the camera was calibrated "
           "at %dx%d." % (w, h, cam.width, cam.height))
    if not allow_scaling:
        raise CalibrationError(
            msg + "\nK is in pixels, so it is only valid at its calibration "
                  "resolution. Re-shoot at %dx%d, or pass allow_scaling=True "
                  "to rescale K (approximate: assumes a pure resize, no crop)."
            % (cam.width, cam.height))

    scaled = cam.scaled_to((w, h))
    warning = (msg + " Intrinsics were rescaled by (%.4f, %.4f). This assumes a "
               "pure resize with no crop; results are approximate."
               % (w / float(cam.width), h / float(cam.height)))
    return scaled, warning


def _validate_inputs(p1, p2, Z_mm):
    if Z_mm is None or not np.isfinite(Z_mm) or Z_mm <= 0:
        raise ValueError("Z must be a positive, finite distance in mm (got %r)" % (Z_mm,))
    if (float(p1[0]), float(p1[1])) == (float(p2[0]), float(p2[1])):
        raise ValueError("the two selected points are identical; pick two distinct points")


# --------------------------------------------------------------------------
# Core primitive: backprojection
# --------------------------------------------------------------------------
def backproject(u, v, Z_mm, cam, undistort=True):
    """Pixel (u, v) at known depth Z -> 3D point (X, Y, Z) in mm, camera frame.

    With undistort=True (default, Option 1) the point is corrected with
    cv2.undistortPoints, which returns normalized coordinates x = X/Z,
    y = Y/Z -- so X = Z*x directly, with NO second division by f_x.
    With undistort=False the raw pinhole inverse is used, which is the exact
    equation quoted in the report:  X = (u - c_x) * Z / f_x.
    """
    Z = float(Z_mm)
    if undistort:
        pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
        norm = cv2.undistortPoints(pts, cam.K, cam.dist).reshape(2)
        return np.array([norm[0] * Z, norm[1] * Z, Z], dtype=np.float64)

    X = (float(u) - cam.cx) * Z / cam.fx
    Y = (float(v) - cam.cy) * Z / cam.fy
    return np.array([X, Y, Z], dtype=np.float64)


def measure(p1, p2, Z_mm, cam, undistort=True):
    """Real-world dimensions between two pixels on a plane at depth Z.

    Returns a dict with:
        length_mm   Euclidean distance between the two backprojected points
                    (the diagonal, i.e. the true dimension for any orientation)
        width_mm    |dX| -- the horizontal component
        height_mm   |dY| -- the vertical component
        du_px/dv_px the pixel span, kept so the result is auditable
        p3d_1/p3d_2 the two 3D points in mm
        mm_per_px   scale at this depth, handy as a sanity check

    Assumes both points lie on a single fronto-parallel plane at depth Z.
    Tilt or object thickness violates that; see the report's limitations.
    """
    _validate_inputs(p1, p2, Z_mm)

    P1 = backproject(p1[0], p1[1], Z_mm, cam, undistort=undistort)
    P2 = backproject(p2[0], p2[1], Z_mm, cam, undistort=undistort)

    d = P2 - P1
    width_mm = abs(float(d[0]))
    height_mm = abs(float(d[1]))
    length_mm = float(math.hypot(d[0], d[1]))

    du_px = abs(float(p2[0]) - float(p1[0]))
    dv_px = abs(float(p2[1]) - float(p1[1]))
    span_px = float(math.hypot(du_px, dv_px))

    return {
        "length_mm": length_mm,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "du_px": du_px,
        "dv_px": dv_px,
        "span_px": span_px,
        "Z_mm": float(Z_mm),
        "p1_px": (float(p1[0]), float(p1[1])),
        "p2_px": (float(p2[0]), float(p2[1])),
        "p3d_1_mm": P1.tolist(),
        "p3d_2_mm": P2.tolist(),
        "mm_per_px": (length_mm / span_px) if span_px > 0 else float("nan"),
        "undistorted_points": bool(undistort),
        # Naive pinhole forms, for the report's side-by-side comparison.
        "width_mm_pinhole": du_px * float(Z_mm) / cam.fx,
        "height_mm_pinhole": dv_px * float(Z_mm) / cam.fy,
    }


def measure_uncertainty(result, cam, sigma_click_px=2.0, sigma_Z_mm=5.0,
                        sigma_f_rel=0.002):
    """First-order error budget for one measurement (plan section 12).

        dW/W ~= d(du)/du + dZ/Z + df_x/f_x

    Click noise on two independent points contributes sqrt(2)*sigma_click.
    Terms are combined in quadrature (independent sources) and also reported
    individually so the report can say which one dominates.
    """
    span = result["span_px"]
    if span <= 0:
        return None
    rel_click = (math.sqrt(2.0) * sigma_click_px) / span
    rel_Z = sigma_Z_mm / result["Z_mm"]
    rel_f = sigma_f_rel
    rel_total = math.sqrt(rel_click ** 2 + rel_Z ** 2 + rel_f ** 2)
    return {
        "rel_click": rel_click,
        "rel_Z": rel_Z,
        "rel_f": rel_f,
        "rel_total": rel_total,
        "abs_total_mm": rel_total * result["length_mm"],
        "dominant": max((rel_click, "click precision"), (rel_Z, "Z measurement"),
                        (rel_f, "focal length"))[1],
    }


# --------------------------------------------------------------------------
# Annotated output (plan section 9)
# --------------------------------------------------------------------------
def annotate(image, p1, p2, result, label=None):
    """Draw the two points, the connecting line and the dimension onto a copy."""
    out = image.copy()
    h, w = out.shape[:2]
    scale = max(0.5, min(w, h) / 1000.0)
    thick = max(2, int(round(scale * 2)))

    a = (int(round(p1[0])), int(round(p1[1])))
    b = (int(round(p2[0])), int(round(p2[1])))

    cv2.line(out, a, b, (0, 215, 255), thick, cv2.LINE_AA)
    for pt in (a, b):
        cv2.circle(out, pt, max(4, thick * 3), (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(out, pt, max(4, thick * 3), (255, 255, 255), thick, cv2.LINE_AA)

    text = label or ("%.1f mm" % result["length_mm"])
    mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
    org = (max(10, mid[0] + 12), max(28, mid[1] - 12))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, text, org, font, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(out, text, org, font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    lines = [
        "Z = %.1f mm" % result["Z_mm"],
        "span = %.1f px  (du=%.1f, dv=%.1f)" % (result["span_px"],
                                                result["du_px"], result["dv_px"]),
        "W = %.1f mm   H = %.1f mm" % (result["width_mm"], result["height_mm"]),
    ]
    y = max(30, int(30 * scale))
    for line in lines:
        cv2.putText(out, line, (10, y), font, scale * 0.6, (0, 0, 0),
                    thick + 2, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), font, scale * 0.6, (255, 255, 255),
                    max(1, thick - 1), cv2.LINE_AA)
        y += int(30 * scale)
    return out


# --------------------------------------------------------------------------
# Self-test -- runs with no photos and no calibration file.
# --------------------------------------------------------------------------
def _self_test():
    """Round-trip check: project a known 3D point, backproject it, compare."""
    print("geometry.py self-test")
    K = np.array([[3000.0, 0.0, 2016.0],
                  [0.0, 3000.0, 1512.0],
                  [0.0, 0.0, 1.0]])
    dist = np.zeros((1, 5))
    cam = Camera(K, dist, (4032, 3024), square_size_mm=25.0)

    # A 300 mm x 200 mm rectangle at Z = 2500 mm, projected by hand.
    Z, W_true, H_true = 2500.0, 300.0, 200.0
    u1 = cam.fx * (-W_true / 2) / Z + cam.cx
    v1 = cam.fy * (-H_true / 2) / Z + cam.cy
    u2 = cam.fx * (W_true / 2) / Z + cam.cx
    v2 = cam.fy * (H_true / 2) / Z + cam.cy

    r = measure((u1, v1), (u2, v2), Z, cam)
    ok = True
    for name, got, want in (("width", r["width_mm"], W_true),
                            ("height", r["height_mm"], H_true),
                            ("diagonal", r["length_mm"], math.hypot(W_true, H_true))):
        good = abs(got - want) < 1e-6
        ok &= good
        print("  %-9s %10.6f mm  expected %10.6f  %s"
              % (name, got, want, "OK" if good else "FAIL"))

    # Undistorted path must agree with the raw pinhole path when dist == 0.
    r_raw = measure((u1, v1), (u2, v2), Z, cam, undistort=False)
    agree = abs(r_raw["length_mm"] - r["length_mm"]) < 1e-6
    ok &= agree
    print("  %-9s undistort=True vs False differ by %.2e mm  %s"
          % ("paths", abs(r_raw["length_mm"] - r["length_mm"]),
             "OK" if agree else "FAIL"))

    # Guards must fire.
    for label, fn in (
        ("Z <= 0", lambda: measure((0, 0), (10, 10), 0.0, cam)),
        ("Z negative", lambda: measure((0, 0), (10, 10), -5.0, cam)),
        ("identical points", lambda: measure((7, 9), (7, 9), 100.0, cam)),
    ):
        try:
            fn()
            print("  guard %-18s FAIL (no exception)" % label)
            ok = False
        except ValueError:
            print("  guard %-18s OK (rejected)" % label)

    # Resolution guard must refuse a mismatched image.
    try:
        check_resolution(cam, (1920, 1080))
        print("  guard %-18s FAIL (no exception)" % "resolution")
        ok = False
    except CalibrationError:
        print("  guard %-18s OK (refused)" % "resolution")

    scaled, warn = check_resolution(cam, (2016, 1512), allow_scaling=True)
    half_ok = abs(scaled.fx - 1500.0) < 1e-9 and abs(scaled.cx - 1008.0) < 1e-9
    ok &= half_ok and warn is not None
    print("  %-9s half-res rescale fx=%.1f cx=%.1f  %s"
          % ("scaling", scaled.fx, scaled.cx, "OK" if half_ok else "FAIL"))

    print("\n%s" % ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
