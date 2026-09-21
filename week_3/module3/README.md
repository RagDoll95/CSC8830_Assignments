# Module 3 - Image Blurring in Spatial and Fourier Domains

This module implements image blurring with a normalized box or Gaussian
filter. It computes the result twice: direct convolution in the spatial domain,
and multiplication in the Fourier domain followed by the inverse Fourier
transform. The comparison numbers and difference image validate that the two
outcomes are the same apart from floating-point numerical precision.

## Web application

From the repository root:

```bash
pip install -r requirements.txt
python app/app.py --host 0.0.0.0
```

Open `http://127.0.0.1:5000/module-3/`, upload an image, select the filter and
parameters, and press **Blur and compare**.

## Command line

```bash
python week_3/module3/image_blur.py photograph.jpg \
  --filter gaussian --size 9 --sigma 2.0 --output module3_output
```

The output folder contains `spatial.png`, `fourier.png`, `difference.png`, and
`comparison.json`.

## Files

- `image_blur.py`: kernels, both filtering paths, comparison, and CLI.
- `theory.md`: assignment theory using only the two supplied lecture decks.
- `test_image_blur.py`: numerical validation on synthetic images.

Both implementations use zero values beyond the border. The Fourier path pads
the image and kernel to the full convolution size before multiplication, then
crops the result to the original image size. This makes the boundary treatment
match the spatial path.
