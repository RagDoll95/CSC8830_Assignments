# Part D — The Two-Camera Relationship

**CSc 8830: Computer Vision — Module 2**

**Question.** Derive mathematically the relationship between the image coordinates of a
point `P = (X, Y, Z)` in camera 1 and in camera 2, where camera 1 is static and camera 2
sits at some distance and at an oblique orientation from camera 1. `P` lies within the
field of view of both cameras. State and justify the assumptions, and clarify all static
parameters, all variables, and how each is computed or determined.

> This document is typed, not photographed, as the submission requires.

---

## 1. Assumptions, and why each is defensible

Each assumption is stated with the reason it holds for this specific setup rather than as
a generic disclaimer.

1. **Ideal pinhole projection.** Both images are undistorted first, using the distortion
   coefficients recovered in Step 1. With an RMS reprojection error below 0.5 px after
   correction, the residual departure from an ideal pinhole is smaller than the
   corner-localization noise, so modelling it further would add parameters without
   reducing error.

2. **Both cameras are calibrated.** `K₁` and `K₂` are known from Step 1's Zhang
   calibration. If a single phone captures both views, then `K₁ = K₂ = K`, which halves
   the unknown intrinsics and is worth stating explicitly — it is the common case here.

3. **Zero skew and square pixels.** The Step 1 fit gives `f_y/f_x` within a fraction of a
   percent of 1, and modern CMOS sensors have orthogonal pixel axes by construction, so
   the skew term in `K` is set to zero rather than estimated.

4. **Rigid relative pose.** `R` and `t` are constant for the pair of exposures. If the two
   views come from a rig, the rig does not flex; if from one phone moved by hand, `R` and
   `t` refer to the two specific exposures and to nothing else.

5. **World frame anchored to camera 1.** This is a free choice of gauge, not a physical
   claim, and it is the choice that makes the algebra clean: camera 1's extrinsics become
   `[I | 0]`, so `P` expressed in the world frame *is* `P` expressed in camera 1's frame.

6. **`P` lies in front of both cameras.** Positive depth in both frames — the chirality
   condition. This is implied by "within the FoV of both," and it is what makes the
   projective equalities below correspond to real image points rather than to points
   behind a camera.

7. **Static scene.** `P` does not move between the two exposures. Required whenever the
   two views are captured sequentially with one camera rather than simultaneously with
   two.

---

## 2. Setting up the two projections

Using the standard projection `x ≅ K [R | t] X`, with `≅` denoting equality up to a
nonzero scale (homogeneous coordinates), and with the world frame on camera 1:

```
Camera 1:    x₁  ≅  K₁ [I | 0] X   =  K₁ P
Camera 2:    x₂  ≅  K₂ [R | t] X   =  K₂ (R P + t)
```

Here:

- `P = (X, Y, Z)ᵀ` is the scene point, nonhomogeneous, in camera 1's frame.
- `x₁ = (u₁, v₁, 1)ᵀ` and `x₂ = (u₂, v₂, 1)ᵀ` are homogeneous image points.
- `R ∈ SO(3)` is camera 2's rotation relative to camera 1.
- `t = −R C̃`, where `C̃` is the camera 2 centre expressed in camera 1's frame. The minus
  sign and the rotation are what make `t` a translation *in camera 2's frame*, not a
  displacement in camera 1's — a standard point of confusion worth naming.

"Oblique orientation" is precisely the statement that `R ≠ I`; "at some distance" is the
statement that `t ≠ 0`. Both matter below.

---

## 3. Back out `P` from camera 1

From the first equation, with `Z` the depth of `P` in camera 1's frame:

```
P  =  Z · K₁⁻¹ x₁
```

This is exactly the backprojection used in Step 2 — the same operation, reused. It also
makes the central difficulty explicit: `x₁` determines the *ray* on which `P` lies, and
nothing more. The scalar `Z` is the missing information.

---

## 4. Substitute, and the answer

Substituting into camera 2's projection:

```
x₂  ≅  K₂ R (Z K₁⁻¹ x₁)  +  K₂ t
    =  Z (K₂ R K₁⁻¹) x₁  +  K₂ t
```

Dividing through by `Z`, which is legal because the relation is projective:

```
        ┌─────────────────────────────────────┐
        │   x₂  ≅  H∞ x₁  +  e₂ / Z           │
        └─────────────────────────────────────┘

where   H∞  =  K₂ R K₁⁻¹        the infinite homography
        e₂  =  K₂ t             the epipole in image 2
```

**This is the required relationship.** In words: the image of `P` in camera 2 is the
position of `x₁` warped by rotation and by the two sets of intrinsics, then displaced
along the direction of the epipole `e₂` by an amount inversely proportional to the depth
of `P`.

Two consequences are worth drawing out:

- `H∞` is the limit as `Z → ∞`. Points at infinity map by `H∞` alone — their images are
  unaffected by the baseline, which is why distant scenery barely shifts between two
  views while nearby objects shift a lot.
- Sweeping `Z` from `0` to `∞` traces a line in image 2: the **epipolar line** of `x₁`.
  A single view therefore cannot recover depth — the whole line is consistent with the
  one observation `x₁`. This is exactly why Step 2 required a tape measure.

---

## 5. The depth-free form: the epipolar constraint

Eliminating the unknown `Z` gives a relation between `x₁` and `x₂` alone. Geometrically,
`P₂ = R P + t` means the three vectors `P₂`, `t`, and `R P` are coplanar, so the triple
product vanishes:

```
P₂ · (t × R P)  =  0
```

Writing this in normalized coordinates and rearranging into matrix form:

```
        ┌─────────────────────────────────────┐
        │   x₂ᵀ F x₁  =  0                    │
        └─────────────────────────────────────┘

where   F  =  K₂⁻ᵀ [t]ₓ R K₁⁻¹      the fundamental matrix
        E  =  [t]ₓ R  =  K₂ᵀ F K₁    the essential matrix
```

and `[t]ₓ` is the skew-symmetric matrix with `[t]ₓ y = t × y`.

Degrees of freedom, which explain what can and cannot be recovered from correspondences:

| Quantity | DoF | Why |
|---|---|---|
| `F` | 7 | 3×3, defined up to scale (−1), and `det F = 0` (−1) |
| `E` | 5 | 3 for `R`, 3 for `t`, minus 1 for the unrecoverable scale of `t` |

Because `E` is unchanged by scaling `t`, **translation is recoverable only up to scale**
from image correspondences alone. Metric scale requires one real length in the world: a
checkerboard square, or a tape-measured baseline `‖t‖`.

---

## 6. Expanded scalar form

Writing the rows of `R` as `r₁ᵀ, r₂ᵀ, r₃ᵀ` and `t = (t_x, t_y, t_z)ᵀ`, camera 2's
projection in scalar form is:

```
u₂  =  f_x2 · (r₁ᵀP + t_x) / (r₃ᵀP + t_z)  +  c_x2

v₂  =  f_y2 · (r₂ᵀP + t_y) / (r₃ᵀP + t_z)  +  c_y2
```

The denominator `r₃ᵀP + t_z` is the depth of `P` in camera 2's frame. This form makes two
things concrete that the matrix form hides: the mapping is **nonlinear** in `P` because of
the perspective division, and the nonlinearity is entirely in that one shared denominator.

Parametrizing `R` by ZYZ Euler angles `(ψ, θ, φ)` ties the rotation to three named
scalars:

```
R(ψ, θ, φ)  =  R_z(φ) R_y(θ) R_z(ψ)
```

so the complete set of static unknowns relating the two cameras is
`(ψ, θ, φ, t_x, t_y, t_z)` — six numbers, of which five are recoverable from
correspondences and the sixth (the scale of `t`) needs one external length.

---

## 7. Parameter table

| Symbol | Type | Meaning | How obtained |
|---|---|---|---|
| `K₁, K₂` | Static | Intrinsic matrix of each camera | Zhang calibration, Step 1. Equal if one phone takes both views. |
| `dist₁, dist₂` | Static | Radial/tangential distortion coefficients | Step 1; applied to undistort both images before any of the above is used. |
| `R` | Static | Relative rotation, camera 1 → camera 2 | `cv2.stereoCalibrate` with the board visible in both views; or `cv2.recoverPose` / decomposition of `E`. |
| `t` | Static | Relative translation, `t = −R C̃` | Same as `R`. From `E` alone it is direction-only; metric scale needs one known length. |
| `‖t‖` | Static | Baseline | Measured directly with a ruler, or fixed by the known board square size in `stereoCalibrate`. |
| `ψ, θ, φ` | Static | ZYZ Euler angles parametrizing `R` | Extracted from `R` by `cv2.Rodrigues` and an Euler conversion. |
| `H∞` | Derived | Infinite homography `K₂ R K₁⁻¹` | Computed from `K₁, K₂, R`. |
| `e₂` | Derived | Epipole in image 2, `K₂ t` | Computed from `K₂, t`. |
| `F` | Derived | Fundamental matrix | From `K₁, K₂, R, t` as above; or estimated directly from ≥ 8 correspondences (`cv2.findFundamentalMat`, 7 with the nonlinear method). |
| `E` | Derived | Essential matrix, `[t]ₓ R` | `K₂ᵀ F K₁`, or `cv2.findEssentialMat` from ≥ 5 correspondences. |
| `x₁, x₂` | Variable | Image coordinates of `P` in each view | Measured: feature detection and matching, or manual selection. |
| `P = (X,Y,Z)` | Variable | The scene point | Unknown; recovered by triangulation once `x₁, x₂, R, t` are known. |
| `Z` | Variable | Depth of `P` in camera 1's frame | Triangulated from the two views, or measured with a tape as in Step 3. |

---

## 8. Degenerate and special cases

**Pure rotation, `t = 0`.** The relation collapses exactly to

```
x₂  ≅  H∞ x₁
```

with no depth term at all. Every point maps by one homography regardless of how far away
it is, so depth is unrecoverable, and `F` is undefined (`E = [0]ₓ R = 0`). This is the
sharpest argument that a nonzero baseline is essential: panning a phone on a tripod, no
matter how many frames are taken, yields no depth information.

**`P` on a plane.** If all points lie on a plane `nᵀP = d`, then `Z` can be eliminated
using the plane equation, and the mapping becomes a plane-induced homography
`H = K₂(R + t nᵀ/d)K₁⁻¹`, again independent of per-point depth. This is the same
coplanar-point degeneracy that makes a single checkerboard view insufficient to calibrate
a camera, and it is why Zhang's method needs the board in *several* orientations.

**`P` on the baseline.** Then `P` projects to the epipole in both images, `x₁ ≅ e₁` and
`x₂ ≅ e₂`, and the epipolar constraint `x₂ᵀ F x₁ = 0` is satisfied identically. The
correspondence carries no information, and triangulation is singular.

---

## 9. Inverting the derivation: triangulation

The payoff of the two-camera formulation is that it removes Step 2's dependence on a
manually measured `Z`. With `x₁, x₂, R, t` and both intrinsics known, the two projection
equations

```
x₁ ≅ K₁[I | 0] X        x₂ ≅ K₂[R | t] X
```

are two constraints on the three unknowns of `P` — plus a third from the second view's
second coordinate, giving four equations for three unknowns. Writing each as a cross
product `x × (PX) = 0` yields a homogeneous linear system `A X = 0` solved by the singular
vector of `A` with the smallest singular value; `cv2.triangulatePoints` does exactly this.

Real-world dimensions then follow as the Euclidean distance between two triangulated
points, with no tape measure anywhere in the pipeline. That is the direct resolution of
the main limitation of the experimental part of this assignment: Step 2's accuracy is
bounded below by how well `Z` can be measured by hand, and Step 3's error statistics show
the resulting bias explicitly. The cost is that metric scale must still enter once, via
the baseline `‖t‖` or a known length in the scene — depth from two views is otherwise
determined only up to a single global scale factor.
