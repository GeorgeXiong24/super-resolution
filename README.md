# Classical Algorithmic Super-Resolution

This project enlarges a single image using only deterministic numerical image
processing. It contains no learned models, neural networks, model downloads,
or external image services. The default reconstruction combines an explicit
camera model, iterative optimisation, and edge-aware regularisation to produce
a sharper, more stable enlargement than interpolation alone.

## What changed in this version

The original residual update has been replaced with a physically consistent
adjoint back-projection step. In particular, the reconstruction now:

- operates on luminance rather than independently altering RGB channels, which
  avoids colour halos at strong edges;
- uses the exact transpose of the implemented Gaussian blur and area-sampling
  operators to return residuals to the high-resolution grid;
- uses reflect boundaries throughout, including at the transpose step, so no
  pixels wrap from one image edge to the opposite edge;
- applies edge-aware Huber total variation in smooth areas while greatly
  reducing smoothing across detected boundaries; and
- applies a small, bounded final detail lift only where local edge evidence
  supports it.

The result remains constrained by the input image: it cannot invent scene
detail that is absent from the observation.

## Reconstruction method

For input luminance `y`, the code models image formation as:

```text
y = D(Gσ(x)) + n
```

where `x` is the unknown high-resolution luminance, `Gσ` is a Gaussian
point-spread-function blur, `D` is integer-factor area sampling, and `n` is
unmodelled capture noise. Lanczos interpolation supplies the initial estimate.
Each iteration then applies the gradient-style correction:

```text
x ← x + μ Gσᵀ Dᵀ (y - D(Gσ(x)))
```

`Dᵀ` distributes a low-resolution residual over the corresponding high
resolution block. `Gσᵀ` is computed with the exact reflect-boundary transpose
of the blur implementation. This gives a mathematically matched forward and
backward model instead of merely enlarging the residual with interpolation.

After the data-consistency update, an edge-aware Huber-TV step regularises the
luminance. Its weights come from the initial luminance gradients: smoothing is
strong in flat/noisy regions and weak across real edges. Cb and Cr chrominance
planes use high-quality Lanczos resampling, then are combined with the refined
luminance plane. A final optional detail lift is edge-limited and deliberately
small to avoid ringing.

`app.py` retains the public function name `ibp_btv_super_resolve` for
compatibility with earlier callers; its implementation is now the improved
adjoint-IBP plus edge-aware Huber-TV pipeline described above.

## Installation

Python 3.9 or newer is required.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The runtime dependencies are NumPy and Pillow. `pillow-heif` is optional at
runtime but included in `requirements.txt` to enable HEIC, HEIF, and AVIF input
when the platform supports it.

## Command-line usage

Run the command from this project directory:

```bash
.venv/bin/python app.py path/to/photo.jpg
```

The default method creates a 2× image beside the source, named
`photo_x2.jpg`. Use `-o` to select a different destination.

```bash
# Higher-quality 4× reconstruction
.venv/bin/python app.py photo.jpg --scale 4 --output photo_4x.png

# Fast, single-pass Lanczos enlargement for comparison
.venv/bin/python app.py photo.jpg --method lanczos --scale 2

# Noisy source: preserve stronger boundaries and slightly increase smoothing
.venv/bin/python app.py photo.jpg --lambda-btv 0.12 --edge-threshold 9

# Disable the optional final detail lift
.venv/bin/python app.py photo.jpg --detail-amount 0
```

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `-s`, `--scale` | `2` | Integer enlargement factor, at least 2. |
| `-m`, `--method` | `ibp-btv` | `ibp-btv` for the reconstruction pipeline, or `lanczos` for fast interpolation only. |
| `-i`, `--iterations` | `20` | Number of data-consistency and regularisation iterations. |
| `--lambda-ibp` | `1.0` | Back-projection step size. |
| `--sigma-psf` | `0.85` | Blur width of the assumed point-spread function, in output-pixel units. |
| `--lambda-btv` | `0.08` | Edge-aware Huber-TV smoothing strength. Higher values suppress more noise but may flatten fine texture. |
| `--edge-threshold` | `12` | Luminance difference treated as an edge. Lower values preserve more boundaries. |
| `--detail-amount` | `0.12` | Final edge-detail lift from 0 (off) to 1. |
| `-o`, `--output` | derived from input | Destination image path. |
| `-q`, `--quality` | `95` | JPEG/WebP output quality. |

Use `--help` to display the same information from the installed program.

## Python API

```python
import numpy as np
from app import ibp_btv_super_resolve

rgb_low_resolution: np.ndarray  # uint8 or floating-point, shape (height, width, 3)
rgb_high_resolution = ibp_btv_super_resolve(
    rgb_low_resolution,
    scale=2,
    iterations=20,
    lambda_ibp=1.0,
    sigma_psf=0.85,
    lambda_btv=0.08,
    edge_threshold=12.0,
    detail_amount=0.12,
)
```

The function returns a floating-point RGB array in the `[0, 255]` range. Clip
and convert to `uint8` when writing with another image library.

## Input and output

Pillow reads common single-frame image formats including JPEG, PNG, BMP, WebP,
TIFF, and GIF. HEIC, HEIF, and AVIF are available when `pillow-heif` can be
installed. Inputs are converted to RGB; transparency and animation frames are
not retained. JPEG input is written as JPEG, while HEIC/HEIF/AVIF input is
written as JPEG unless `--output` specifies another Pillow-supported format.

For best results, start with a clean, reasonably focused image and save the
result as PNG or high-quality JPEG. This method improves sampling and edge
rendering under its stated model; it does not reconstruct unknown objects or
recover information removed by severe blur, compression, or sensor saturation.
