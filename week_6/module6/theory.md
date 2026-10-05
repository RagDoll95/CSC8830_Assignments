# Motion tracking equations

Brightness constancy: I(x+dx,y+dy,t+dt)=I(x,y,t).
For small motion, Taylor expansion gives Ix dx+Iy dy+It dt≈0. Divide by dt:

**Ix u+Iy v+It=0**, where u=dx/dt, v=dy/dt.

One pixel gives one equation with two unknowns. Lucas-Kanade assumes constant motion
inside a small window. Stack rows [Ix,Iy] in A and values -It in b. Least squares gives:

(AᵀA)d=Aᵀb,

AᵀA=[[ΣIx²,ΣIxIy],[ΣIxIy,ΣIy²]], Aᵀb=[-ΣIxIt,-ΣIyIt]ᵀ.

Solve for d=(dx,dy); the next position is (x+dx,y+dy). Textured corners give independent
constraints; smooth areas and straight edges are unreliable. For larger displacement,
solve at coarse pyramid resolution, scale the estimate up and refine it, as in the slides.

## Bilinear interpolation

For x=m+α,y=n+β with m=floor(x),n=floor(y):

I(x,y)=(1-α)(1-β)I(m,n)+α(1-β)I(m+1,n)
      +(1-α)βI(m,n+1)+αβI(m+1,n+1).

This interpolates horizontally in two rows, then vertically between them.
For neighboring values [10,20;30,40] and α=.25,β=.5, the result is 22.5.
Subpixel tracking uses interpolation because predicted locations need not be integers.

## Worked examples

`report.pdf` contains the complete derivations, measured-pixel comparisons for both
videos, numerical normal equations, and a four-view planar reconstruction.
`measurements.csv` contains independent visual annotations in the 640x360 images.
They are approximate (about 1–2 pixels), not exact ground truth. The beak example
is retained as a failure case. Run `validate.py` to reproduce all calculations.

Part B is explicitly **simulated**, with orthographic cameras, known scale, and a
front-parallel first view. It uses centering and rank-2 SVD from the SfM lecture,
then fixes the planar metric using that known first camera. It does not recover
unknown general camera poses or nonzero depth. Run `sfm.py` for its images, camera
parameters, matrices, and boundary reconstruction. Real captured views and their
camera measurements must be supplied if the instructor requires photographs.

References: Nayar (2025), *Optical Flow*, FPCV-4-3 (Lucas-Kanade and coarse-to-fine
sections), and *Structure from Motion*, FPCV-4-4, pp. 2–15.
