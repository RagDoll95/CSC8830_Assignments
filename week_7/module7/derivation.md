# Part 2 — Uncalibrated stereo derivation study notes

These are typed study notes, not the final submission. Assignment 7 specifically requires this part handwritten and scanned. Work through the derivation and write it in your own notation, including the assumptions and citations.

## 1. Assumptions and coordinate convention

Nayar's uncalibrated stereo lecture assumes **known intrinsics** K_L and K_R, but unknown relative rotation and translation [2, pp. 2–3]. Undistort the observed pixel coordinates first. A static scene must have enough reliable, nondegenerate correspondences (at least eight for the linear method; preferably spread across different depths). Pure camera rotation provides no triangulation baseline. An entirely planar match set is degenerate for unrestricted fundamental-matrix recovery.

Use the left camera as the world frame. Define X_R = R X_L + t, so the projection matrices are P_L=K_L[I|0], P_R=K_R[R|t]. Here t is the left origin expressed in the right frame; the right camera center in the left frame is C_R=−Rᵀt.

Below the epipolar equation is **p_Rᵀ F p_L=0**. Nayar uses the transposed convention p_Lᵀ F_lecture p_R=0, so F=F_lectureᵀ. Keep the convention consistent; the recovered physical geometry is the same.

## 2. Derive the epipolar constraint

For homogeneous image points p_L=(u_L,v_L,1)ᵀ and p_R=(u_R,v_R,1)ᵀ, let a=K_L⁻¹p_L and b=K_R⁻¹p_R. Their third components are one. For depths λ and μ:

    X_L = λa
    X_R = μb = R(λa) + t.

The three vectors b, Ra and t lie in the same epipolar plane. Their scalar triple product is zero:

    bᵀ(t × Ra) = 0.

Define the skew-symmetric cross-product matrix

    [t]× = [[0, −tz, ty], [tz, 0, −tx], [−ty, tx, 0]].

Then E=[t]×R and bᵀEa=0. Substituting a and b gives

    p_Rᵀ K_R⁻ᵀ E K_L⁻¹ p_L = 0,
    F = K_R⁻ᵀ E K_L⁻¹,
    E = K_Rᵀ F K_L.

This is the lecture's epipolar geometry expressed with the right-to-left transpose convention [2, pp. 4–8].

## 3. Estimate F from corresponding pixels

Expanding one correspondence gives

    uR*uL*f11 + uR*vL*f12 + uR*f13
    + vR*uL*f21 + vR*vL*f22 + vR*f23
    + uL*f31 + vL*f32 + f33 = 0.

Thus a row of A is

    [uR*uL, uR*vL, uR, vR*uL, vR*vL, vR, uL, vL, 1].

Stack at least eight independent rows and solve Af=0 subject to ||f||=1. If A=UΣVᵀ, f is the last column of V (equivalently the smallest-eigenvalue eigenvector of AᵀA). Reshape f into F. This is the lecture's constrained least-squares construction [2, pp. 9–10].

For numerical stability, first center each set of image points and scale its mean distance from the origin to √2, using transforms T_L,T_R. Solve for F_n from normalized coordinates. Enforce rank two by replacing the smallest singular value of F_n with zero, then denormalize F=T_Rᵀ F_n T_L. Reject bad matches rather than trust every pair. These are numerical refinements of the linear derivation.

## 4. Recover pose and establish metric scale

Compute E=K_Rᵀ F K_L. Let E=U diag(s1,s2,s3) Vᵀ. Enforce the essential-matrix structure with singular values (s,s,0), where s=(s1+s2)/2. Choose signs of the zero-singular-value columns so U and V are proper rotations. Set

    W = [[0,−1,0], [1,0,0], [0,0,1]].

The candidate poses are R=UWVᵀ or UWᵀVᵀ, and t direction = ±U[:,3] (third column). Choose the candidate for which triangulated points are in front of both cameras. E fixes the **direction**, not the physical magnitude, of t [2, pp. 9–10, scale ambiguity].

If baseline B is measured in mm, set t=B*t_hat where ||t_hat||=1. Alternatively triangulate at unit baseline and use a known reference length L_ref:

    s = L_ref / ||X_ref2_hat − X_ref1_hat||,
    X_i = s X_i_hat,     t = s t_hat.

Without B or a known reference distance, the reconstruction has an arbitrary overall scale and cannot yield millimetres. If intrinsics are also unknown, general two-view reconstruction has projective ambiguity, so one reference length alone does not make it metric; extra calibration or metric constraints are needed.

## 5. Recover each endpoint's depth

From μb=λRa+t, rearrange

    G q = t,    G = [−Ra  b],    q = [λ, μ]ᵀ.

For noisy rays, use least squares q=G⁺t, computed by SVD. If G has full column rank, equivalently q=(GᵀG)⁻¹Gᵀt. Both depths must be positive. Because a_z=1, λ is the left camera's axial depth Z. Then

    X_L = λa = (λa_x, λa_y, λ)ᵀ.

This recovers depth first, then location, as required [2, pp. 13–14]. Sparse endpoint correspondences suffice for measurement; dense correspondence is only needed for a full surface reconstruction.

## 6. Calculate dimensions

For every measured edge joining reconstructed endpoints i and j:

    L_ij = sqrt((Xi−Xj)^2 + (Yi−Yj)^2 + (Zi−Zj)^2).

Rectangle vertices 1,2,3,4 in boundary order: width=L12, height=L23; compare L34 and L41 as opposite-side consistency checks. Circle: use two true opposite diameter endpoints; diameter=L12. Polygon: report L12,L23,...,Ln1. Each endpoint can have a different depth; a single common Z is not assumed.

## 7. Worked numerical example and calibrated special case

Take K_L=K_R=diag(800,800,1), principal point (0,0), R=I, and a measured baseline B=200 mm. The right camera lies at (200,0,0), so t=(−200,0,0). Consequently

    E = [[0,0,0], [0,0,200], [0,−200,0]],
    F = K⁻ᵀ E K⁻¹ = [[0,0,0], [0,0,0.25], [0,−0.25,0]].

The epipolar equation becomes 0.25(v_R−v_L)=0, i.e. matching rows. Consider four corresponding corners:

| Corner | Left (u,v) | Right (u,v) |
|---|---|---|
| 1 | (0,0) | (−80,0) |
| 2 | (160,0) | (80,0) |
| 3 | (160,80) | (80,80) |
| 4 | (0,80) | (−80,80) |

Coordinates here are principal-point-centered, so negative values are allowed. For corner 2, a=(0.2,0,1), b=(0.1,0,1). The ray equations give

    −0.2λ + 0.1μ = −200,
    −λ + μ = 0,
    μ = λ,  −0.1λ = −200,
    λ = μ = 2000 mm.

Hence corner 2 is (400,0,2000). All four disparities are d=80 px, so for each point Z=800*200/80=2000 mm. The reconstructed corners are

    X1=(0,0,2000), X2=(400,0,2000),
    X3=(400,200,2000), X4=(0,200,2000).

Width=sqrt(400²)=400 mm; height=sqrt(200²)=200 mm. Opposite edges agree. A diameter between X1 and X3 would be sqrt(400²+200²)=447.214 mm **only if those were physically opposite points on a circle**; this rectangle's diagonal is not automatically a circle diameter.

This example verifies the triangulation and scale steps after pose recovery; its four coplanar points are not enough to estimate an unrestricted F. General pose recovery must use the nondegenerate correspondence set from Step 3. In the parallel-camera case, subtracting u_R=fx(X−B)/Z+cx from u_L=fx*X/Z+cx immediately gives d=fx*B/Z and Z=fx*B/d, the calibrated method implemented for Parts 1 and 3 [1, p. 12].

## References

[1] Nayar, S. K. (2025). *Camera Calibration*. FPCV-4-1, Columbia University, pp. 12–13. https://cave.cs.columbia.edu/Statics/monographs/Camera%20Calibration%20FPCV-4-1.pdf

[2] Nayar, S. K. (2025). *Uncalibrated Stereo*. FPCV-4-2, Columbia University, pp. 2–14. https://cave.cs.columbia.edu/Statics/monographs/Uncalibrated%20Stereo%20FPCV-4-2.pdf

[3] Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision* (2nd ed.), Cambridge University Press, chapters 9, 11 and 12 (normalized eight-point estimation, essential-matrix decomposition and triangulation). Listed as further reading in the lecture.
