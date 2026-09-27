# super-resolution

A lightweight command-line tool that doubles the width and height of an image using a simple 2x nearest-neighbor-aware interpolation.

## How it works

For each row and column, the algorithm inserts one new pixel between every pair of existing pixels. The new pixel value is the average of its two neighbors. The last pixel is mirrored by averaging the final two pixels of the original axis.

The interpolation is applied first along the horizontal axis, then along the vertical axis, producing an image with twice the original width and height.

## Setup

1. Make sure Python 3 is installed.
2. Clone or download this repository.
3. Open a terminal in the project folder and create a virtual environment:

   ```bash
   python -m venv .venv
   ```

4. Activate the virtual environment and install dependencies:

   ```bash
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

   > `.venv/` is listed in `.gitignore` because it contains system-specific files and should not be committed. Each user creates their own local copy.

## Usage

After completing the Setup steps above, the `.venv` folder exists on your local machine, so you can run:

```bash
.venv\Scripts\python.exe app.py
```

When prompted, enter the full path to the image you want to upscale, for example:

```
D:\Pictures\photo.jpg
```

The result is saved next to the input file as `<filename>_2x.<ext>`.

## Supported input formats

Common formats such as JPEG, PNG, BMP, WebP, TIFF, GIF, and HEIC/HEIF/AVIF (with `pillow-heif`) are supported through Pillow.

## Note

This is a basic interpolation-based upscaler, not a machine-learning super-resolution model. It is fast and deterministic, but will not recover real detail that is not present in the original image.
