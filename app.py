import os
import sys

import numpy as np
from PIL import Image

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pillow_heif = None


def double_axis(arr, axis):
    """沿 axis 轴做 2 倍插值。

    规则：
    - 每个像素的右/下插入一个新像素，新像素 RGB = 左右(上下)相邻两像素的平均值；
    - 末尾像素的右/下也插入一个新像素，取值 = 末尾像素与其左/上相邻像素的平均（镜像补边）。
    """
    arr = np.asarray(arr)
    n = arr.shape[axis]

    # 把目标轴移到第 0 维，方便切片
    moved = np.moveaxis(arr, axis, 0)
    out = np.zeros((2 * n,) + moved.shape[1:], dtype=np.uint8)

    # 偶数位放置原始像素
    out[0::2] = moved

    # 奇数位：中间插值 = 相邻两像素的平均
    out[1::2][: n - 1] = (
        (moved[:-1].astype(np.int32) + moved[1:].astype(np.int32)) // 2
    ).astype(np.uint8)

    # 末尾补边：最后一个新像素 = 末像素与倒数第二个原像素的平均
    out[2 * n - 1] = (
        (moved[n - 1].astype(np.int32) + moved[n - 2].astype(np.int32)) // 2
    ).astype(np.uint8)

    # 移回原轴
    return np.moveaxis(out, 0, axis)


def main():
    path = input("Enter the path of the image to upscale: ").strip().strip('"').strip("'")
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        sys.exit(1)

    ext = os.path.splitext(path)[1].lower()

    # HEIC/HEIF/AVIF are read via pillow-heif but written out as jpg here
    if ext in (".heic", ".heif", ".avif"):
        if pillow_heif is None:
            print("Reading HEIC/HEIF/AVIF requires pillow-heif: pip install pillow-heif")
            sys.exit(1)
        out_ext = ".jpg"
    elif ext == ".jpeg":
        out_ext = ".jpg"
    else:
        out_ext = ext

    dst = f"{os.path.splitext(path)[0]}_2x{out_ext}"

    # Read every pixel's RGB into `img`, shape (height, width, 3), dtype uint8
    img = np.array(Image.open(path).convert("RGB"))

    img = double_axis(img, axis=1)  # horizontal: double width
    img = double_axis(img, axis=0)  # vertical: double height

    save_kwargs = {"quality": 95} if out_ext == ".jpg" else {}
    Image.fromarray(img).save(dst, **save_kwargs)
    print(f"Saved: {dst}  (size {img.shape[1]}x{img.shape[0]})")


if __name__ == "__main__":
    main()
