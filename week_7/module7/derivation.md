# Part 2 — Uncalibrated stereo derivation study notes

These are typed study notes, not the final submission. Assignment 7 requires this part handwritten and scanned. Work through the derivation and write it out in your own hand, including the assumptions and citations.

The derivation follows the notation and conventions of the two lectures [1, 2]. Page numbers are the printed monograph pages. One step, the explicit decomposition of E, is stated but not derived in the lecture; it is marked **Beyond the lecture** and uses only elementary vector algebra.

## 1. Assumptions and conventions

The intrinsics of both cameras are known; their relative rotation and translation are not [2, pp. 2–3]:

    K_l = [[fx_l, 0, ox_l], [0, fy_l, oy_l], [0, 0, 1]],   K_r likewise.

Two views of a static scene, at least eight reliable correspondences (u_l, v_l) ↔ (u_r, v_r) [2, p. 3], and one measured length for metric scale (section 6).

x_l and x_r are the same scene point in the left and right camera frames. t is the position of the right camera in the left camera's frame; R is the orientation of the left camera in the right camera's frame [2, p. 6]:

    x_l = R x_r + t.                                                    (0)

## 2. Essential matrix

The camera centres and the scene point span the epipolar plane. Its normal is n = t × x_l, which is perpendicular to x_l [2, p. 5]:

    x_l · (t × x_l) = 0.

Write the cross product as a skew-symmetric matrix:

    T× = [[0, −tz, ty], [tz, 0, −tx], [−ty, tx, 0]],   t × a = T× a.

Substitute (0) for the second x_l. Because T× t = t × t = 0 [2, p. 6]:

    x_lᵀ T× (R x_r + t) = x_lᵀ T× R x_r = 0,
    x_lᵀ E x_r = 0,   E = T× R.                                       (1)

## 3. Fundamental matrix

Perspective projection gives z_l (u_l, v_l, 1)ᵀ = K_l x_l and z_r (u_r, v_r, 1)ᵀ = K_r x_r [2, p. 7]. Hence x_lᵀ = z_l ũ_lᵀ K_l⁻ᵀ and x_r = z_r K_r⁻¹ ũ_r. Substitute into (1):

    z_l z_r ũ_lᵀ K_l⁻ᵀ E K_r⁻¹ ũ_r = 0.

The scene is in front of both cameras, so z_l ≠ 0 and z_r ≠ 0; divide them out [2, p. 7]:

    ũ_lᵀ F ũ_r = 0,   F = K_l⁻ᵀ E K_r⁻¹,   E = K_lᵀ F K_r.             (2)

## 4. Estimate F

Expanding (2) for correspondence i gives one linear equation [2, p. 9]:

    (f11 u_r + f12 v_r + f13) u_l + (f21 u_r + f22 v_r + f23) v_l + f31 u_r + f32 v_r + f33 = 0.

So each match contributes the row

    a_i = [u_l u_r, u_l v_r, u_l, v_l u_r, v_l v_r, v_l, u_r, v_r, 1].

Stack the rows into A (m × 9) to get A f = 0. F acts on homogeneous coordinates, so F and kF describe the same geometry. Fix the scale with ‖f‖² = 1. That leaves eight unknowns, so at least eight matches are needed [2, p. 9].

Minimise ‖A f‖² subject to ‖f‖² = 1 with L(f, λ) = fᵀAᵀA f − λ(fᵀf − 1). Setting ∂L/∂f = 0 gives AᵀA f = λ f. At such an f, ‖A f‖² = λ, so f is the eigenvector of AᵀA with the smallest eigenvalue [2, p. 10; 1, p. 9]. Rearrange f into F.

## 5. Recover t and R from E

Compute E = K_lᵀ F K_r [2, p. 10]. It has the arbitrary scale (and sign) of F. The lecture states that T× and R can be decoupled from E using SVD [2, pp. 6, 10].

**Beyond the lecture.** The following derivation uses only facts the lectures give (T× is skew-symmetric, R R^T = I [1, p. 5], t × t = 0) plus standard cross-product identities.

*Translation.* Since R Rᵀ = I, E Eᵀ = T× T×ᵀ. Multiplying out:

    T× T×ᵀ = [[ty²+tz², −tx ty, −tx tz], [−tx ty, tx²+tz², −ty tz], [−tx tz, −ty tz, tx²+ty²]]
           = ‖t‖² I − t tᵀ.

Its trace is 2‖t‖², so rescale E until tr(E Eᵀ) = 2. Then ‖t‖ = 1, written t̂, and

    t̂ t̂ᵀ = I − E Eᵀ.

Take the column with the largest diagonal entry and divide it by that entry's square root. Equivalently, t̂ is the eigenvector of E Eᵀ with eigenvalue 0, because E Eᵀ t̂ = t̂ − t̂ (t̂ᵀ t̂) = 0.

*Rotation.* Let e_i be the columns of E and r_i the columns of R. From E = T× R, e_i = t̂ × r_i. Define w_i = e_i × t̂. Then

    w_i = (t̂ × r_i) × t̂ = r_i − (t̂ · r_i) t̂,

which is the part of r_i perpendicular to t̂. For (i, j, k) cyclic, w_j × w_k is parallel to t̂ with length t̂ · (r_j × r_k) = t̂ · r_i, using r_j × r_k = r_i for a rotation. Therefore

    r_1 = w_1 + w_2 × w_3,   r_2 = w_2 + w_3 × w_1,   r_3 = w_3 + w_1 × w_2.

*Choosing the solution.* The signs of E and t̂ are both unknown, which gives four (R, t) candidates. Keep the one that puts a triangulated point in front of both cameras, z_l > 0 and z_r > 0 [2, p. 7].

## 6. Metric scale

If the world and the stereo system are both scaled by the same factor, the images do not change [2, p. 9; 1, p. 9]. Two images therefore give t, and every length, only up to scale. Fix the scale with one physical measurement:

- **Measured baseline:** tape-measure the camera displacement B and set t = B t̂.
- **Reference length:** triangulate with t = t̂. If two points with known separation L_ref reconstruct ℓ_ref apart, multiply every coordinate by s = L_ref / ℓ_ref.

Without either, uncalibrated stereo gives shape and proportions but not millimetres.

## 7. Depth by triangulation

The left imaging equation becomes ũ_l ≡ P_l x̃_r with P_l = K_l [R | t]. The right one is ũ_r ≡ M_int_r x̃_r with M_int_r = [K_r | 0] [2, p. 14]. Cross-multiplying the four ratio equations gives four linear equations in three unknowns [2, p. 15]:

    [ u_r m31 − m11   u_r m32 − m12   u_r m33 − m13 ]           [ m14 − u_r m34 ]
    [ v_r m31 − m21   v_r m32 − m22   v_r m33 − m23 ] [x_r]     [ m24 − v_r m34 ]
    [ u_l p31 − p11   u_l p32 − p12   u_l p33 − p13 ] [y_r]  =  [ p14 − u_l p34 ]
    [ v_l p31 − p21   v_l p32 − p22   v_l p33 − p23 ] [z_r]     [ p24 − v_l p34 ]

    A x_r = b   ⇒   x_r = (AᵀA)⁻¹ Aᵀ b.

The depth is z_r. The left-camera depth z_l is the third component of R x_r + t. If an object corner is hard to match, search along its epipolar line (a, b, c) = ũ_lᵀ F, which reduces correspondence to a 1D search [2, pp. 11–13].

## 8. Object dimensions

Express all endpoints in the same frame:

    L_ij = sqrt((x_i − x_j)² + (y_i − y_j)² + (z_i − z_j)²).

- **Rectangle**, corners 1–4 in order: width = L12 and height or length = L23. L34 and L41 are consistency checks.
- **Circle:** two diametrically opposite physical points give diameter = L12.
- **Polygon:** L12, L23, …, Ln1.

Each corner has its own depth; no equal-depth assumption is used.

## 9. Check: the calibrated special case

With R = I and t = (b, 0, 0)ᵀ [1, p. 12]:

    E = [[0, 0, 0], [0, 0, −b], [0, b, 0]],   x_lᵀ E x_r = b (z_l y_r − y_l z_r) = 0.

With z_l = z_r this gives v_l = v_r, the scan-line constraint [1, p. 13]. Subtracting u_r − o_x = f_x (x − b)/z from u_l − o_x = f_x x/z gives

    z = b f_x / (u_l − u_r),

which is the calibrated method implemented for Parts 1 and 3 [1, p. 12].

## 10. Worked example

K_l = K_r = [[1000, 0, 640], [0, 1000, 360], [0, 0, 1]]. The camera move is unknown except for its tape-measured length, B = 400 mm. Pixels are given to 0.01 px. Corners 1–4 are a rectangle; points 5–8 are other scene features, so the eight points are not coplanar.

| i | u_l | v_l | u_r | v_r |
|---|---|---|---|---|
| 1 | 690.00 | 260.00 | 483.38 | 264.46 |
| 2 | 885.76 | 275.25 | 728.60 | 275.29 |
| 3 | 885.76 | 444.75 | 728.60 | 444.71 |
| 4 | 690.00 | 460.00 | 483.38 | 455.54 |
| 5 | 640.00 | 467.14 | 542.71 | 463.51 |
| 6 | 973.33 | 193.33 | 710.78 | 193.94 |
| 7 | 733.75 | 360.00 | 666.33 | 360.00 |
| 8 | 947.69 | 398.46 | 823.33 | 399.10 |

1. **One row of A.** For match 1: a_1 = [333532.2, 182477.4, 690, 125678.8, 68759.6, 260, 483.38, 264.46, 1].
2. **F.** The eigenvector solve is numerical. Scaled to f33 = 1:

       F ≈ [[5.698e−9, 1.777e−8, −1.238e−5], [3.755e−6, 2.204e−8, 1.819e−2], [−1.354e−3, −2.096e−2, 1]].

   Check: the epipolar line of corner 1 is (a, b, c) = (−3.74e−4, −2.0945e−2, 5.7199). At u_r = 483.38 it gives v_r = 264.46, which matches.
3. **E and t̂.** E = KᵀFK, normalised to tr(EEᵀ) = 2 and rounded to 4 d.p.:

       E ≈ [[0, 0, 0], [0.1793, 0, 0.9836], [0, −1.0002, 0]],
       I − EEᵀ ≈ diag(1, 0.0004, −0.0004),   so t̂ = ±(1, 0, 0).

4. **R.** Using −E and t̂ = (1, 0, 0): w_1 = (0, 0, 0.1793), w_2 = (0, 1.0002, 0), w_3 = (0, 0, 0.9836). Then

       R ≈ [[0.9838, 0, −0.1793], [0, 1.0002, 0], [0.1793, 0, 0.9836]],

   a rotation of about 10.3° about y.
5. **Choose the candidate.** At unit baseline, corner 1 gives (z_r, z_l) = (+3.50, −3.35) for (+E, +t̂), (−2.62, −2.51) for (+E, −t̂), (+2.62, +2.51) for (−E, +t̂) and (−3.50, +3.35) for (−E, −t̂). Only (−E, +t̂) has both depths positive.
6. **Scale.** t = 400 t̂ = (400, 0, 0) mm.
7. **Triangulate corner 1.**

       P_l = K[R | t] = [[1098.55, 0, 450.17, 400000], [64.55, 1000.2, 354.10, 0], [0.1793, 0, 0.9836, 0]],
       A = [[−1000, 0, −156.62], [0, −1000, −95.54], [−974.83, 0, 228.52], [−17.93, −1000.2, −98.36]],
       b = (0, 0, 400000, 0)ᵀ.

   Solving gives x_r = (−164.35, −100.25, 1049.33) mm and x_l = R x_r + t = (50.13, −100.27, 1002.66) mm, so the depth is Z ≈ 1.00 m from the left camera.
8. **Corners 2–4.** Repeating the same steps gives depths z_r of 1184.12, 1184.12 and 1049.33 mm. The rectangle is tilted.
9. **Size.** L12 = sqrt(269.26² + 0.04² + 134.79²) ≈ 301.1 mm and L23 ≈ 200.6 mm. The generating geometry was 300 × 200 mm. Most of the 0.4% error comes from rounding E to 4 d.p.; carried at full precision, the result is 300.17 × 200.08 mm.

## References

[1] Nayar, S. K. (2025). *Camera Calibration*. FPCV-4-1, Columbia University. https://cave.cs.columbia.edu/Statics/monographs/Camera%20Calibration%20FPCV-4-1.pdf

[2] Nayar, S. K. (2025). *Uncalibrated Stereo*. FPCV-4-2, Columbia University. https://cave.cs.columbia.edu/Statics/monographs/Uncalibrated%20Stereo%20FPCV-4-2.pdf
