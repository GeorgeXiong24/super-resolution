"""
Advanced single-image super-resolution without machine learning.

Core algorithm
--------------
Iterative Back Projection (IBP) with Bilateral Total Variation (BTV)
regularization.

* IBP enforces the physical consistency constraint: if the reconstructed
  high-resolution image is blurred by the estimated Point Spread Function (PSF)
  and down-sampled, it should reproduce the observed low-resolution image.
* BTV regularization suppresses ringing and noise amplification while
  preserving sharp edges by penalising intensity differences weighted by
  spatial distance.

A high-quality Lanczos separable interpolation is used for the initial
estimate and for propagating the residual back to the high-resolution grid.
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np
from PIL import Image

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pillow_heif = None


# ---------------------------------------------------------------------------
# Utility kernels and convolution
# ---------------------------------------------------------------------------

def _lanczos_kernel(x: np.ndarray, a: int = 3) -> np.ndarray:
    """Lanczos windowed sinc kernel, vectorised."""
    x = np.abs(x).astype(np.float64)
    out = np.zeros_like(x)
    zero = x < 1e-9
    inside = (x < a) & ~zero
    out[zero] = 1.0
    x_in = x[inside]
    out[inside] = (
        a * np.sin(np.pi * x_in) * np.sin(np.pi * x_in / a)
        / (np.pi ** 2 * x_in ** 2)
    )
    return out


def _gaussian_kernel(sigma: float, size: Optional[int] = None) -> np.ndarray:
    """Create a normalised 1-D Gaussian kernel."""
    if size is None:
        size = max(7, int(6 * sigma) | 1)  # odd, at least 7
    if size % 2 == 0:
        size += 1
    r = np.arange(size) - size // 2
    k = np.exp(-(r ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _convolve_separable(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply a separable convolution with reflect padding."""
    if img.ndim == 2:
        img = img[..., None]
        squeeze = True
    else:
        squeeze = False

    pad = len(kernel) // 2
    h, w, c = img.shape
    tmp = np.empty((h, w, c), dtype=np.float64)
    out = np.empty((h, w, c), dtype=np.float64)

    kernel = kernel.astype(np.float64)

    # Horizontal pass
    for i in range(h):
        for ch in range(c):
            row = np.pad(img[i, :, ch], pad, mode="reflect")
            tmp[i, :, ch] = np.convolve(row, kernel, mode="same")[pad:-pad]

    # Vertical pass
    for j in range(w):
        for ch in range(c):
            col = np.pad(tmp[:, j, ch], pad, mode="reflect")
            out[:, j, ch] = np.convolve(col, kernel, mode="same")[pad:-pad]

    return out[..., 0] if squeeze else out


# ---------------------------------------------------------------------------
# Interpolation and resampling
# ---------------------------------------------------------------------------

def _lanczos_weights(src_size: int, dst_size: int, a: int = 3) -> np.ndarray:
    """
    Build a (dst_size, src_size) weight matrix for 1-D Lanczos resampling.
    Each row is normalised to sum to 1.
    """
    scale = dst_size / src_size
    weights = np.zeros((dst_size, src_size), dtype=np.float64)
    for i in range(dst_size):
        # Pixel centre in source coordinate space
        u = (i + 0.5) / scale - 0.5
        left = int(np.floor(u - a + 1))
        right = int(np.ceil(u + a))
        support = np.arange(left, right + 1)
        valid = (support >= 0) & (support < src_size)
        support = support[valid]
        vals = _lanczos_kernel(u - support, a)
        s = vals.sum()
        if s > 0:
            vals /= s
        weights[i, support] = vals
    return weights


def lanczos_resize(img: np.ndarray, scale: float, a: int = 3) -> np.ndarray:
    """
    Resize ``img`` by ``scale`` using separable Lanczos resampling.
    Returns float64 array in [0, 255] range preserving channel count.
    """
    img = np.asarray(img, dtype=np.float64)
    squeeze = False
    if img.ndim == 2:
        img = img[..., None]
        squeeze = True

    h, w, c = img.shape
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    W = _lanczos_weights(w, new_w, a)
    H = _lanczos_weights(h, new_h, a)

    # Horizontal then vertical: out = H @ (img @ W.T)
    out = np.empty((new_h, new_w, c), dtype=np.float64)
    for ch in range(c):
        horiz = img[:, :, ch] @ W.T          # (h, new_w)
        out[:, :, ch] = H @ horiz            # (new_h, new_w)

    if squeeze:
        out = out[..., 0]
    return np.clip(out, 0.0, 255.0)


def downsample_area(img: np.ndarray, scale: int) -> np.ndarray:
    """
    Integer-scale area downsampling by averaging non-overlapping scale x scale
    blocks.
    """
    img = np.asarray(img, dtype=np.float64)
    squeeze = False
    if img.ndim == 2:
        img = img[..., None]
        squeeze = True

    h, w, c = img.shape
    new_h, new_w = h // scale, w // scale
    img = img[: new_h * scale, : new_w * scale]
    blocks = img.reshape(
        new_h, scale, new_w, scale, c
    ).transpose(0, 2, 4, 1, 3).reshape(new_h, new_w, c, scale * scale)
    out = blocks.mean(axis=-1)
    return out[..., 0] if squeeze else out


# ---------------------------------------------------------------------------
# Iterative Back Projection + BTV regularization
# ---------------------------------------------------------------------------

def _btv_regularize(
    img: np.ndarray,
    lambda_btv: float,
    alpha_btv: float,
    neighbors: int = 1,
) -> np.ndarray:
    """
    One gradient-descent step of Bilateral Total Variation regularization.

    Minimises  sum_{(l,m) in neighborhood} alpha^{|l|+|m|}
               || img - shift(img, (l,m)) ||_1
    """
    out = img.copy()
    total_weight = 0.0
    for dy in range(-neighbors, neighbors + 1):
        for dx in range(-neighbors, neighbors + 1):
            if dx == 0 and dy == 0:
                continue
            weight = alpha_btv ** (abs(dx) + abs(dy))
            total_weight += weight
            shifted = np.roll(img, (dy, dx), axis=(0, 1))
            out -= lambda_btv * weight * np.sign(img - shifted)
    if total_weight > 0:
        out /= 1.0 + lambda_btv * total_weight
    return out


def ibp_btv_super_resolve(
    lr: np.ndarray,
    scale: int,
    iterations: int = 25,
    lambda_ibp: float = 1.2,
    sigma_psf: float = 1.0,
    lambda_btv: float = 0.04,
    alpha_btv: float = 0.7,
    btv_neighbors: int = 1,
) -> np.ndarray:
    """
    Single-image super-resolution by Iterative Back Projection with BTV
    regularisation.

    Parameters
    ----------
    lr : np.ndarray
        Low-resolution image, uint8 or float in [0, 255].
    scale : int
        Integer upscaling factor.
    iterations : int
        Number of IBP iterations.
    lambda_ibp : float
        Step size for the back-projected residual.
    sigma_psf : float
        Standard deviation of the Gaussian Point Spread Function used to
        simulate the imaging blur.
    lambda_btv : float
        Strength of the BTV regularisation term.
    alpha_btv : float
        Spatial damping factor for the BTV weights (0 < alpha < 1).
    btv_neighbors : int
        Maximum shift distance included in the BTV neighbourhood.
    """
    lr = np.asarray(lr, dtype=np.float64)
    hr = lanczos_resize(lr, scale, a=3)
    psf_kernel = _gaussian_kernel(sigma_psf)

    for _ in range(iterations):
        # Simulate observed low-resolution image
        blurred = _convolve_separable(hr, psf_kernel)
        lr_sim = downsample_area(blurred, scale)

        # Residual in LR space, back-projected to HR space
        residual_lr = lr - lr_sim
        residual_hr = lanczos_resize(residual_lr, scale, a=2)

        # Update
        hr = hr + lambda_ibp * residual_hr

        # BTV regularisation
        hr = _btv_regularize(
            hr, lambda_btv=lambda_btv, alpha_btv=alpha_btv,
            neighbors=btv_neighbors,
        )

        hr = np.clip(hr, 0.0, 255.0)

    return hr


# ---------------------------------------------------------------------------
# Simple interpolation baseline
# ---------------------------------------------------------------------------

def bicubic_resize(img: np.ndarray, scale: float) -> np.ndarray:
    """
    Pillow's built-in bicubic resize, exposed as a numpy-based baseline.
    """
    mode = "L" if img.ndim == 2 else "RGB"
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode=mode)
    new_h = max(1, int(round(img.shape[0] * scale)))
    new_w = max(1, int(round(img.shape[1] * scale)))
    resized = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float64)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_output_path(input_path: str, scale: int, ext: Optional[str]) -> str:
    base, in_ext = os.path.splitext(input_path)
    out_ext = ext if ext is not None else in_ext
    return f"{base}_x{scale}{out_ext}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Advanced non-AI single-image super-resolution."
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to the image to upscale. If omitted, a prompt is shown.",
    )
    parser.add_argument(
        "-s", "--scale", type=int, default=2,
        help="Integer upscaling factor (default: 2).",
    )
    parser.add_argument(
        "-m", "--method", choices=["ibp-btv", "lanczos"], default="ibp-btv",
        help=(
            "Super-resolution method. 'ibp-btv' (default) uses Iterative Back "
            "Projection with Bilateral Total Variation regularisation; "
            "'lanczos' uses high-quality Lanczos resampling only."
        ),
    )
    parser.add_argument(
        "-i", "--iterations", type=int, default=25,
        help="IBP iterations (default: 25).",
    )
    parser.add_argument(
        "--lambda-ibp", type=float, default=1.2,
        help="IBP residual step size (default: 1.2).",
    )
    parser.add_argument(
        "--sigma-psf", type=float, default=1.0,
        help="Gaussian PSF sigma simulating the imaging blur (default: 1.0).",
    )
    parser.add_argument(
        "--lambda-btv", type=float, default=0.04,
        help="BTV regularisation strength (default: 0.04).",
    )
    parser.add_argument(
        "--alpha-btv", type=float, default=0.7,
        help="BTV spatial damping factor (default: 0.7).",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output path. Default is '<input>_x<scale>.<ext>'.",
    )
    parser.add_argument(
        "-q", "--quality", type=int, default=95,
        help="JPEG/WebP output quality (default: 95).",
    )
    args = parser.parse_args()

    if args.image:
        path = args.image
    else:
        path = input("Enter the path of the image to upscale: ").strip().strip('"').strip("'")

    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    ext = os.path.splitext(path)[1].lower()
    if ext in (".heic", ".heif", ".avif"):
        if pillow_heif is None:
            print(
                "Reading HEIC/HEIF/AVIF requires pillow-heif: "
                "pip install pillow-heif",
                file=sys.stderr,
            )
            return 1
        out_ext = ".jpg"
    elif ext == ".jpeg":
        out_ext = ".jpg"
    else:
        out_ext = ext

    img = np.array(Image.open(path).convert("RGB"))
    print(
        f"Input: {path} ({img.shape[1]}x{img.shape[0]}), scale={args.scale}, "
        f"method={args.method}"
    )

    if args.method == "lanczos":
        result = lanczos_resize(img, args.scale)
    else:
        result = ibp_btv_super_resolve(
            img,
            scale=args.scale,
            iterations=args.iterations,
            lambda_ibp=args.lambda_ibp,
            sigma_psf=args.sigma_psf,
            lambda_btv=args.lambda_btv,
            alpha_btv=args.alpha_btv,
        )

    result = np.clip(result, 0, 255).astype(np.uint8)

    dst = args.output or _resolve_output_path(path, args.scale, out_ext)

    save_kwargs: dict = {}
    if out_ext in (".jpg", ".jpeg", ".webp"):
        save_kwargs["quality"] = args.quality
    Image.fromarray(result).save(dst, **save_kwargs)
    print(f"Saved: {dst} ({result.shape[1]}x{result.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
