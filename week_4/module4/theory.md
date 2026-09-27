# Fourier-domain edge detection and region segmentation

Let f(x,y) be image intensity and define the Fourier transform using angular
frequencies:

$$F(\omega_x,\omega_y)=\iint f(x,y)e^{-i(\omega_x x+\omega_y y)}\,dx\,dy.$$

## 1. Why derivatives can be computed in the frequency domain

Integration by parts in x gives

$$\mathcal F\{f_x\}=\int [f e^{-i\omega_x x}]_{-\infty}^{\infty}e^{-i\omega_y y}\,dy
+i\omega_x\iint f e^{-i(\omega_x x+\omega_y y)}\,dx\,dy
=i\omega_x F.$$

Here the boundary term vanishes for a decaying signal. Finite images require a
chosen extension/padding convention. Similarly, F{f_y}=i omega_y F. Repeating
this argument gives F{f_xx}=−omega_x² F and F{f_yy}=−omega_y² F.

The logic: a derivative measures rapid spatial change. Multiplication by
frequency suppresses the constant (zero-frequency) component and strengthens
rapid variations. Noise also has high-frequency content, so smoothing matters.

## 2. Gaussian smoothing and gradient edges

For a normalized spatial Gaussian of standard deviation sigma,

$$H_\sigma(\omega_x,\omega_y)=e^{-\sigma^2(\omega_x^2+\omega_y^2)/2}.$$

The convolution theorem gives L=G_sigma*f=F^−1{H_sigma F}. Its derivatives are

$$L_x=\mathcal F^{-1}\{i\omega_x H_\sigma F\},\qquad
L_y=\mathcal F^{-1}\{i\omega_y H_\sigma F\}.$$

Compute gradient magnitude M=sqrt(L_x²+L_y²) and direction atan2(L_y,L_x).
Edges are large local maxima of M along the gradient direction. Nonmaximum
suppression and hysteresis thresholds can then link strong responses while
retaining connected weaker responses. These final operations are nonlinear
spatial decisions; one frequency multiplier does not implement all of Canny.

The logic: smoothing reduces noise, differentiation reveals transitions,
and thinning/thresholding chooses the locations that become the edge map.
Our code uses OpenCV's spatial Canny implementation; this derivation explains
the frequency-domain equivalent of its smoothing/derivative stages.

## 3. Laplacian edges

Adding the two second derivatives yields

$$\nabla^2 L=\mathcal F^{-1}\{-(\omega_x^2+\omega_y^2)H_\sigma F\}.$$

Candidate edges lie at zero crossings of this Laplacian-of-Gaussian response.
Require a sufficient response change across the crossing to suppress noise.
The Laplacian is scalar and does not supply the gradient's edge direction.
A zero crossing is a boundary cue, not proof of an object boundary.

## 4. Region segmentation using filtered responses

For two regions with distinct mean intensities, use the model

$$f=\mu_b+(\mu_f-\mu_b)\mathbf1_R+\eta,$$

where R is the foreground and eta is noise. Frequency-domain low-pass filtering
and reconstruction give

$$s=\mathcal F^{-1}\{H_\sigma F\}
=\mu_b+(\mu_f-\mu_b)(G_\sigma*\mathbf1_R)+G_\sigma*\eta.$$

Well inside a sufficiently large foreground region, G_sigma*1_R is approximately
1; well outside it is approximately 0. Thus s is near mu_f or mu_b while noise
is attenuated. If mu_f>mu_b, equal class priors and equal variances give the
nearest-mean decision rule

$$(s-\mu_f)^2<(s-\mu_b)^2
\iff s>(\mu_f+\mu_b)/2.$$

Thresholding therefore gives a region mask; connected-component analysis groups
adjacent foreground pixels into regions. If the foreground is darker, reverse
the inequality. Near boundaries smoothing mixes regions, so large sigma can
shift or erase fine details. If the intensities overlap, this simple model fails.

For texture-defined regions, apply several band-pass filters H_k and construct
local response energies

$$r_k=\mathcal F^{-1}\{H_kF\},\qquad
E_k=G_\rho*|r_k|^2.$$

Different textures concentrate energy in different frequency bands or
orientations. Thresholding these local energy maps can separate regions with
known texture differences. Squaring and region labeling are nonlinear steps.
Global Fourier magnitudes alone do not locate a region; phase and the inverse
transform retain the spatial information needed to assign pixels.

The logic: filtering makes a chosen region property (intensity or texture)
easier to distinguish; reconstruction returns that evidence to image coordinates;
a decision rule and connectivity produce the regions. Fourier analysis alone
does not identify which region is a human.

## Discrete implementation

Use a 2D DFT/FFT with frequencies omega_x=2*pi*fftfreq(width) and similarly for y.
Keep filters aligned with the unshifted FFT, or shift both consistently. An FFT
implicitly assumes periodic boundaries: pad before filtering and crop afterward
to reduce opposite-edge wraparound. Spectral derivatives describe the periodic
interpolant; exact equivalence to a finite Sobel kernel requires using that
kernel's DFT as the multiplier instead of i*omega.

## Lecture connections

Edge Detection FPCV-2-1: gradients, Laplacian zero crossings, Gaussian smoothing,
and Canny. Boundary Detection FPCV-2-2: edge pixels require grouping into object
boundaries. SIFT Detector FPCV-2-3: Gaussian scale space and differences of
Gaussians connect smoothing scale to the structures retained in an image.
