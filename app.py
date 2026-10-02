"""Deterministic single-image super-resolution based on a regularised inverse problem.

The image formation model blurs a high-resolution candidate with a Gaussian
point-spread function then area-samples it.  Each residual is propagated by
the adjoint of that exact model.  Edge-aware Huber total variation keeps the
optimisation stable without blurring strong boundaries; chroma uses Lanczos
resampling so reconstructed luminance edges do not acquire colour fringes.
"""

from __future__ import annotations

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


def _validate_parameters(scale: int, iterations: int, lambda_ibp: float,
                         sigma_psf: float, lambda_tv: float,
                         edge_threshold: float, detail_amount: float) -> None:
    if scale < 2:
        raise ValueError("scale must be an integer of at least 2")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if lambda_ibp <= 0 or sigma_psf <= 0 or lambda_tv < 0:
        raise ValueError("IBP step and PSF sigma must be positive; TV strength cannot be negative")
    if edge_threshold <= 0:
        raise ValueError("edge_threshold must be positive")
    if not 0 <= detail_amount <= 1:
        raise ValueError("detail_amount must be in the range [0, 1]")


# ---------------------------------------------------------------------------
# Resampling and the image-formation model
# ---------------------------------------------------------------------------

def _lanczos_kernel(x: np.ndarray, a: int = 3) -> np.ndarray:
    """Lanczos (windowed-sinc) interpolation kernel."""
    x = np.abs(np.asarray(x, dtype=np.float64))
    result = np.zeros_like(x)
    at_origin = x < 1e-12
    inside = (x < a) & ~at_origin
    result[at_origin] = 1.0
    xi = x[inside]
    result[inside] = a * np.sin(np.pi * xi) * np.sin(np.pi * xi / a) / (np.pi**2 * xi**2)
    return result


def _reflect_indices(indices: np.ndarray, size: int) -> np.ndarray:
    """Map arbitrary indices onto [0, size) with reflect-101 boundaries."""
    if size == 1:
        return np.zeros_like(indices)
    period = 2 * size - 2
    reflected = np.mod(indices, period)
    return np.where(reflected < size, reflected, period - reflected)


def _lanczos_taps(source_size: int, destination_size: int, a: int = 3) -> tuple[np.ndarray, np.ndarray]:
    destination = np.arange(destination_size, dtype=np.float64)
    coordinate = (destination + 0.5) * source_size / destination_size - 0.5
    offsets = np.arange(-a + 1, a + 1)
    raw_indices = np.floor(coordinate).astype(int)[:, None] + offsets[None, :]
    weights = _lanczos_kernel(coordinate[:, None] - raw_indices, a)
    weights /= weights.sum(axis=1, keepdims=True)
    return _reflect_indices(raw_indices, source_size), weights


def lanczos_resize(img: np.ndarray, scale: int, a: int = 3) -> np.ndarray:
    """Upscale a gray or RGB image with separable Lanczos interpolation."""
    image = np.asarray(img, dtype=np.float64)
    is_gray = image.ndim == 2
    if is_gray:
        image = image[..., None]
    if image.ndim != 3:
        raise ValueError("image must have shape (height, width) or (height, width, channels)")
    height, width, channels = image.shape
    target_height, target_width = height * scale, width * scale
    x_indices, x_weights = _lanczos_taps(width, target_width, a)
    horizontal = np.zeros((height, target_width, channels), dtype=np.float64)
    for tap in range(x_weights.shape[1]):
        horizontal += image[:, x_indices[:, tap], :] * x_weights[None, :, tap, None]
    y_indices, y_weights = _lanczos_taps(height, target_height, a)
    result = np.zeros((target_height, target_width, channels), dtype=np.float64)
    for tap in range(y_weights.shape[1]):
        result += horizontal[y_indices[:, tap], :, :] * y_weights[:, tap, None, None]
    return result[..., 0] if is_gray else result


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(2, int(np.ceil(3.0 * sigma)))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(coordinates**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def _convolve_separable(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Separable reflect-boundary convolution, vectorised across rows and channels."""
    source = np.asarray(image, dtype=np.float64)
    is_gray = source.ndim == 2
    if is_gray:
        source = source[..., None]
    radius = len(kernel) // 2
    height, width, _ = source.shape
    padded_y = np.pad(source, ((radius, radius), (0, 0), (0, 0)), mode="reflect")
    horizontal = np.zeros_like(source)
    for offset, weight in enumerate(kernel):
        horizontal += weight * padded_y[offset:offset + height, :, :]
    padded_x = np.pad(horizontal, ((0, 0), (radius, radius), (0, 0)), mode="reflect")
    result = np.zeros_like(source)
    for offset, weight in enumerate(kernel):
        result += weight * padded_x[:, offset:offset + width, :]
    return result[..., 0] if is_gray else result


def _convolution_axis_adjoint(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Transpose of one reflect-boundary convolution axis."""
    source = np.asarray(image, dtype=np.float64)
    if axis == 1:
        return np.swapaxes(
            _convolution_axis_adjoint(np.swapaxes(source, 0, 1), kernel, 0), 0, 1
        )
    radius = len(kernel) // 2
    length = source.shape[0]
    result = np.zeros_like(source)
    base_indices = np.arange(length)
    for offset, weight in enumerate(kernel):
        source_indices = _reflect_indices(base_indices + offset - radius, length)
        np.add.at(result, source_indices, weight * source)
    return result


def _convolve_separable_adjoint(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Exact transpose of ``_convolve_separable`` for reflect boundaries."""
    source = np.asarray(image, dtype=np.float64)
    is_gray = source.ndim == 2
    if is_gray:
        source = source[..., None]
    # The forward operator applies vertical then horizontal filtering, so its
    # transpose applies the horizontal transpose then the vertical transpose.
    result = _convolution_axis_adjoint(
        _convolution_axis_adjoint(source, kernel, 1), kernel, 0
    )
    return result[..., 0] if is_gray else result


def downsample_area(image: np.ndarray, scale: int) -> np.ndarray:
    """Forward sampling operator: average non-overlapping scale-by-scale blocks."""
    source = np.asarray(image, dtype=np.float64)
    is_gray = source.ndim == 2
    if is_gray:
        source = source[..., None]
    height, width, channels = source.shape
    if height % scale or width % scale:
        raise ValueError("high-resolution dimensions must be divisible by scale")
    result = source.reshape(height // scale, scale, width // scale, scale, channels).mean(axis=(1, 3))
    return result[..., 0] if is_gray else result


def _area_adjoint(residual: np.ndarray, scale: int) -> np.ndarray:
    """Adjoint of area sampling: spread a residual over its contributing block."""
    source = np.asarray(residual, dtype=np.float64)
    return np.repeat(np.repeat(source, scale, axis=0), scale, axis=1) / (scale * scale)


# ---------------------------------------------------------------------------
# Edge-aware reconstruction in luminance space
# ---------------------------------------------------------------------------

def _rgb_to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(rgb, dtype=np.float64)
    red, green, blue = source[..., 0], source[..., 1], source[..., 2]
    y = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = 128.0 - 0.168736 * red - 0.331264 * green + 0.5 * blue
    cr = 128.0 + 0.5 * red - 0.418688 * green - 0.081312 * blue
    return y, cb, cr


def _ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    cb, cr = cb - 128.0, cr - 128.0
    return np.stack((y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb), axis=-1)


def _edge_weights(reference: np.ndarray, edge_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Large smoothing weights in flat areas and small weights across edges."""
    horizontal = 1.0 / (1.0 + (np.abs(np.diff(reference, axis=1)) / edge_threshold) ** 2)
    vertical = 1.0 / (1.0 + (np.abs(np.diff(reference, axis=0)) / edge_threshold) ** 2)
    return horizontal, vertical


def _edge_aware_huber_tv_step(image: np.ndarray, horizontal_weight: np.ndarray,
                               vertical_weight: np.ndarray, strength: float,
                               huber_scale: float = 4.0) -> np.ndarray:
    """One stable anisotropic Huber-TV diffusion step with no boundary wrapping."""
    if strength == 0:
        return image
    dx = image[:, 1:] - image[:, :-1]
    dy = image[1:, :] - image[:-1, :]
    flux_x = horizontal_weight * dx / np.sqrt(dx * dx + huber_scale * huber_scale)
    flux_y = vertical_weight * dy / np.sqrt(dy * dy + huber_scale * huber_scale)
    update = np.zeros_like(image)
    update[:, :-1] += flux_x
    update[:, 1:] -= flux_x
    update[:-1, :] += flux_y
    update[1:, :] -= flux_y
    return image + strength * update


def _edge_limited_detail(image: np.ndarray, amount: float, edge_threshold: float) -> np.ndarray:
    """Mild final detail lift, bounded by a local edge-strength mask."""
    if amount == 0:
        return image
    blurred = _convolve_separable(image, _gaussian_kernel(0.65))
    gradient_x = np.zeros_like(image)
    gradient_y = np.zeros_like(image)
    gradient_x[:, 1:] = np.abs(image[:, 1:] - image[:, :-1])
    gradient_y[1:, :] = np.abs(image[1:, :] - image[:-1, :])
    edge = np.hypot(gradient_x, gradient_y)
    return image + amount * edge / (edge + edge_threshold) * (image - blurred)


def ibp_btv_super_resolve(lr: np.ndarray, scale: int, iterations: int = 20,
                           lambda_ibp: float = 1.0, sigma_psf: float = 0.85,
                           lambda_btv: float = 0.08, alpha_btv: float = 0.7,
                           btv_neighbors: int = 1, edge_threshold: float = 12.0,
                           detail_amount: float = 0.12) -> np.ndarray:
    """Reconstruct an RGB image with adjoint IBP and edge-aware Huber-TV.

    ``alpha_btv`` and ``btv_neighbors`` remain accepted for API compatibility
    but are intentionally unused: data-derived edge weights replace the old
    circularly shifted BTV neighbourhood, eliminating wraparound artifacts.
    """
    _validate_parameters(scale, iterations, lambda_ibp, sigma_psf, lambda_btv, edge_threshold, detail_amount)
    source = np.asarray(lr, dtype=np.float64)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("lr must be an RGB array with shape (height, width, 3)")
    y_lr, cb_lr, cr_lr = _rgb_to_ycbcr(source)
    y_hr = lanczos_resize(y_lr, scale)
    cb_hr, cr_hr = lanczos_resize(cb_lr, scale), lanczos_resize(cr_lr, scale)
    psf = _gaussian_kernel(sigma_psf)
    horizontal_weight, vertical_weight = _edge_weights(y_hr, edge_threshold)
    for _ in range(iterations):
        simulated_lr = downsample_area(_convolve_separable(y_hr, psf), scale)
        residual_lr = y_lr - simulated_lr
        correction = _convolve_separable_adjoint(_area_adjoint(residual_lr, scale), psf)
        y_hr += lambda_ibp * correction
        y_hr = _edge_aware_huber_tv_step(y_hr, horizontal_weight, vertical_weight, lambda_btv)
        y_hr = np.clip(y_hr, 0.0, 255.0)
    y_hr = _edge_limited_detail(y_hr, detail_amount, edge_threshold)
    return np.clip(_ycbcr_to_rgb(y_hr, cb_hr, cr_hr), 0.0, 255.0)


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------

def _resolve_output_path(input_path: str, scale: int, extension: Optional[str]) -> str:
    base, input_extension = os.path.splitext(input_path)
    return f"{base}_x{scale}{extension if extension is not None else input_extension}"


def _read_rgb(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic super-resolution using adjoint IBP and edge-aware Huber-TV.")
    parser.add_argument("image", nargs="?", help="Path to the image to upscale. If omitted, a prompt is shown.")
    parser.add_argument("-s", "--scale", type=int, default=2, help="Integer upscaling factor, at least 2 (default: 2).")
    parser.add_argument("-m", "--method", choices=["ibp-btv", "lanczos"], default="ibp-btv", help="'ibp-btv' (default) or fast 'lanczos'.")
    parser.add_argument("-i", "--iterations", type=int, default=20, help="IBP iterations (default: 20).")
    parser.add_argument("--lambda-ibp", type=float, default=1.0, help="IBP residual step size (default: 1.0).")
    parser.add_argument("--sigma-psf", type=float, default=0.85, help="Gaussian PSF sigma in HR pixels (default: 0.85).")
    parser.add_argument("--lambda-btv", type=float, default=0.08, help="Edge-aware Huber-TV strength (default: 0.08).")
    parser.add_argument("--edge-threshold", type=float, default=12.0, help="Edge-preservation threshold in luma levels (default: 12).")
    parser.add_argument("--detail-amount", type=float, default=0.12, help="Bounded final edge-detail lift from 0 to 1 (default: 0.12).")
    parser.add_argument("-o", "--output", default=None, help="Output path. Default is '<input>_x<scale>.<ext>'.")
    parser.add_argument("-q", "--quality", type=int, default=95, help="JPEG/WebP output quality (default: 95).")
    args = parser.parse_args()
    path = args.image or input("Enter the path of the image to upscale: ").strip().strip('"').strip("'")
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        return 1
    extension = os.path.splitext(path)[1].lower()
    if extension in (".heic", ".heif", ".avif"):
        if pillow_heif is None:
            print("Reading HEIC/HEIF/AVIF requires pillow-heif: pip install pillow-heif", file=sys.stderr)
            return 1
        output_extension = ".jpg"
    else:
        output_extension = ".jpg" if extension == ".jpeg" else extension
    try:
        _validate_parameters(args.scale, args.iterations, args.lambda_ibp, args.sigma_psf, args.lambda_btv, args.edge_threshold, args.detail_amount)
        image = _read_rgb(path)
        if args.method == "lanczos":
            result = lanczos_resize(image, args.scale)
        else:
            result = ibp_btv_super_resolve(image, args.scale, args.iterations, args.lambda_ibp, args.sigma_psf, args.lambda_btv, edge_threshold=args.edge_threshold, detail_amount=args.detail_amount)
    except (OSError, ValueError) as error:
        print(f"Could not process image: {error}", file=sys.stderr)
        return 1
    destination = args.output or _resolve_output_path(path, args.scale, output_extension)
    save_kwargs: dict[str, int] = {}
    if output_extension in (".jpg", ".jpeg", ".webp"):
        save_kwargs["quality"] = args.quality
    Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB").save(destination, **save_kwargs)
    print(f"Saved: {destination} ({result.shape[1]}x{result.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
