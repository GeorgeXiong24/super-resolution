# super-resolution

A command-line single-image super-resolution tool that uses advanced
mathematical and computational algorithms — **without any machine-learning or
AI computation**.

## How it works

The default method is **Iterative Back Projection (IBP)** with **Bilateral
Total Variation (BTV)** regularization. This is a classic, well-founded
computational super-resolution technique.

1. **Initial estimate** — the low-resolution (LR) image is upscaled using
   high-quality separable **Lanczos** interpolation (a windowed-sinc kernel)
   to form the first high-resolution (HR) guess.
2. **Image formation model** — the HR estimate is blurred with an estimated
   **Point Spread Function** (a Gaussian PSF) and down-sampled (area
   averaging) to simulate how the observed LR image was captured.
3. **Back projection** — the difference between the simulated LR image and the
   real LR image (the residual) is projected back onto the HR grid, and the HR
   estimate is corrected. Iterating this enforces the physical consistency
   constraint that the reconstructed image, when degraded, reproduces the
   observation.
4. **BTV regularization** — after each update, a Bilateral Total Variation
   prior is applied. It suppresses ringing artifacts and noise amplification
   while preserving sharp edges by penalizing intensity differences weighted by
   spatial distance.

A plain **Lanczos** resampling method is also available for comparison or when
a fast, single-pass upscale is preferred.

## Setup

1. Make sure Python 3.9+ is installed.
2. Clone or download this repository.
3. Open a terminal in the project folder and create a virtual environment:

   ```bash
   python -m venv .venv
   ```

4. Activate the virtual environment and install dependencies:

   ```bash
   .venv/bin/python -m pip install -r requirements.txt
   ```

   > `.venv/` is listed in `.gitignore` because it contains system-specific
   > files and should not be committed. Each user creates their own local copy.

## Usage

```bash
.venv/bin/python app.py path/to/image.jpg
```

The result is saved next to the input file as `<filename>_x<scale>.<ext>`.

### Options

```
positional arguments:
  image                 Path to the image to upscale. If omitted, a prompt is shown.

options:
  -s, --scale SCALE     Integer upscaling factor (default: 2).
  -m, --method METHOD   'ibp-btv' (default) or 'lanczos'.
  -i, --iterations N    IBP iterations (default: 25).
  --lambda-ibp FLOAT    IBP residual step size (default: 1.2).
  --sigma-psf FLOAT     Gaussian PSF sigma (default: 1.0).
  --lambda-btv FLOAT    BTV regularization strength (default: 0.04).
  --alpha-btv FLOAT     BTV spatial damping factor (default: 0.7).
  -o, --output PATH     Output path.
  -q, --quality N       JPEG/WebP output quality (default: 95).
```

### Examples

```bash
# 4x upscale with the default IBP+BTV method
.venv/bin/python app.py photo.jpg -s 4

# Fast Lanczos-only upscale
.venv/bin/python app.py photo.jpg -m lanczos -s 2

# Stronger regularization to reduce artifacts on noisy images
.venv/bin/python app.py photo.jpg -s 2 --lambda-btv 0.10
```

## Supported input formats

Common formats such as JPEG, PNG, BMP, WebP, TIFF, GIF, and HEIC/HEIF/AVIF
(with `pillow-heif`) are supported through Pillow.

## Note

This tool reconstructs a higher-resolution image using deterministic
mathematical algorithms (IBP + BTV + Lanczos). It does not use neural networks
or learned models, and therefore cannot recover detail that was never captured
in the source image — but it produces sharp, well-conditioned enlargements that
strictly honor the imaging model.
