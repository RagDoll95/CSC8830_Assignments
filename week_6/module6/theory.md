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

## Structure from motion

Orthographic Tomasi-Kanade factorization (FPCV-4-4): W = MS with centroid-subtracted
image coordinates; rank(W) <= 2 for a plane, so W = U1 S1 V1^T with two singular values,
M = U1 S1^(1/2) Q, S = Q^-1 S1^(1/2) V1^T, and Q from |i_f| = |j_f| = 1, i_f . j_f = 0
(the out-of-plane parts of i_f, j_f are extra unknowns), solved by Newton's method.

`answers.pdf` contains the derivations, worked numbers, figures, and limitations.
Run `validate.py` and `sfm.py` to reproduce every number.
