#!/usr/bin/env python3
"""
Grayscale + CLAHE preprocessing (same logic as rape_root_seg/dataset/2.py).

Reads BGR/RGB field photos, writes single-channel CLAHE gray saved as RGB PNG/JPG
(3 identical channels — matches dataset/da/images training style).

  python clahe_preprocess.py --image /path/to/5-1_fan_.JPG --out ./clahe/5-1_fan_.jpg
  python clahe_preprocess.py --input_dir ./img_1 --out_dir ./clahe_out
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def imread_robust(path: str | Path):
    path = str(path)
    try:
        with open(path, "rb") as stream:
            buf = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            return img
        return cv2.imread(path)
    except Exception as e:
        print(f"  [错误] 读取失败 {path}: {e}")
        return None


def clahe_gray_rgb(img_bgr: np.ndarray, clip_limit: float, grid_size: tuple[int, int]) -> np.ndarray:
    if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced = clahe.apply(gray)
    return np.stack([enhanced, enhanced, enhanced], axis=2)


def save_rgb(path: Path, rgb: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        Image.fromarray(rgb).save(path, quality=95)
    else:
        Image.fromarray(rgb).save(path)
    return True


def process_one(src: Path, dst: Path, clip_limit: float, grid_size: tuple[int, int]) -> bool:
    img = imread_robust(src)
    if img is None:
        return False
    rgb = clahe_gray_rgb(img, clip_limit, grid_size)
    save_rgb(dst, rgb)
    print(f"[ok] {src.name} -> {dst}  ({rgb.shape[1]}x{rgb.shape[0]})")
    return True


def main():
    ap = argparse.ArgumentParser(description="Grayscale + CLAHE (2.py compatible)")
    ap.add_argument("--image", type=str, default="", help="Single input image")
    ap.add_argument("--out", type=str, default="", help="Single output path (.jpg/.png)")
    ap.add_argument("--input_dir", type=str, default="", help="Batch input folder")
    ap.add_argument("--out_dir", type=str, default="", help="Batch output folder")
    ap.add_argument("--clip_limit", type=float, default=2.0)
    ap.add_argument("--grid_w", type=int, default=8)
    ap.add_argument("--grid_h", type=int, default=8)
    args = ap.parse_args()
    grid = (args.grid_w, args.grid_h)

    if args.image:
        if not args.out:
            sys.exit("Single mode requires --out")
        ok = process_one(Path(args.image), Path(args.out), args.clip_limit, grid)
        sys.exit(0 if ok else 1)

    if args.input_dir:
        out_dir = Path(args.out_dir or (Path(args.input_dir).name + "_clahe"))
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.JPG", "*.JPEG", "*.PNG"):
            paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
        if not paths:
            sys.exit(f"No images in {args.input_dir}")
        ok = 0
        for p in sorted(paths):
            src = Path(p)
            dst = out_dir / src.name
            if process_one(src, dst, args.clip_limit, grid):
                ok += 1
        print(f"[done] {ok}/{len(paths)} -> {out_dir}")
        return

    ap.print_help()
    sys.exit("Provide --image + --out, or --input_dir")


if __name__ == "__main__":
    main()
