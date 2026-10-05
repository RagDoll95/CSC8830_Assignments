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

## Validation and interpretation

For each 30-second video, compare several corresponding corners in the exported first
and second frames against `points.csv` rows with frame=0. Record measured and predicted
positions and endpoint error sqrt((xpred-xactual)²+(ypred-yactual)²). Independently
measured positions are needed; tracker output alone is not ground truth.

Arrows show apparent direction and displacement in pixels/frame. Multiplying by FPS
gives pixels/second, not real-world speed. Camera motion and lighting changes affect
flow. Cuts, occlusion, blur and low texture can produce failures. Include actual frames
and coordinate examples to support observations in the report.

## Part B: four-view example

Capture four overlapping views of a stationary textured flat object, such as a book
cover. Record image size, object dimensions, camera calibration and camera positions.
The SfM slides center tracked coordinates, construct W (8×N for four views), and use
SVD factorization W≈MS with orthonormality constraints on camera rows. A planar object
has centered rank at most 2, so do not assume unrestricted full-rank 3D reconstruction.
Actual images and their numerical calculations are still needed for this exercise.

References: Nayar (2025), *Optical Flow*, FPCV-4-3 (Lucas-Kanade and coarse-to-fine
sections), and *Structure from Motion*, FPCV-4-4. Lucas & Kanade (1981), IJCAI, 674–679.
