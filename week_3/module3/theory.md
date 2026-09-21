# Module 3 Theory: Spatial Filtering and the Fourier Equivalent

## Image blurring by filtering

Let the discrete image be `f[i,j]` and the filter be `h[i,j]`. The lecture
slides also call `h` a mask or kernel. For each output location, two-dimensional
convolution flips the filter in both directions, overlays it on the image, and
sums the products in the overlap region:

```text
g[i,j] = f[i,j] * h[i,j]
```

The implementation offers a box filter and a Gaussian filter. The weights of
either kernel sum to 1. As shown in *Image Processing I*, increasing the size of
the box filter produces more smoothing. The Gaussian filter also smooths the
image, with nearby pixels contributing more than pixels farther from the
center. In both cases, the blurred image is the result `g`.

The border needs an explicit rule because a finite image does not contain all
pixels covered by a kernel centered near an edge. This implementation uses zero
values beyond the image border in both calculations.

## Convolution theorem

The derivation in *Image Processing II* starts with

```text
g(x) = f(x) * h(x)
```

and takes its Fourier transform. Substituting the convolution integral and
separating the resulting double integral gives one integral equal to `F(u)` and
another equal to `H(u)`. Therefore

```text
G(u) = F(u) H(u).
```

Thus, convolution in the spatial domain is multiplication in the frequency
domain. The equivalent filtering sequence is:

1. Take the Fourier transform of the image to obtain `F`.
2. Take the Fourier transform of the filter to obtain `H`.
3. Multiply them to obtain `G = F H`.
4. Take the inverse Fourier transform of `G` to obtain the blurred image `g`.

The slides describe a Gaussian in the frequency domain as a low-pass filter.
The multiplication reduces the high frequencies, and the inverse Fourier
transform produces a blurred version of the image. A wider Gaussian kernel in
the spatial domain produces more blur.

## Experimental validation

The application applies exactly the same normalized kernel to the same image
using the two sequences below.

| Spatial domain | Frequency domain |
|---|---|
| Convolve image and kernel directly | Transform image and kernel |
| Obtain the blurred image | Multiply the transforms |
| - | Apply the inverse Fourier transform |

For each pixel and color channel it then forms the absolute difference between
the two floating-point results. It reports:

- average absolute difference; and
- maximum absolute difference.

It also displays a scaled difference image. Values close to zero validate the
slide result that spatial convolution and Fourier-domain multiplication give
the same output. The comparison is performed before conversion to an ordinary
8-bit display image, so rounding for display does not hide a disagreement.

## Slide sources

- *First Principles of Computer Vision, Image Processing I (FPCV-1-4)*:
  convolution, two-dimensional discrete convolution, the border problem, box
  filtering, Gaussian filtering, and smoothing.
- *First Principles of Computer Vision, Image Processing II (FPCV-1-5)*:
  Fourier transform, convolution theorem, convolution using the Fourier
  transform, Gaussian smoothing in the Fourier domain, and low-pass filtering.
