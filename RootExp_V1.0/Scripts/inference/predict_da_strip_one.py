#!/usr/bin/env python3
"""
RootExp DA strip inference on one image pair.

Network input: preprocessed DA image (dataset/da/images style).
Visualization overlay: optional original img_1 (same WxH as DA image).

  python predict_da_strip_one.py \
    --da_image /path/to/IMG_0298.jpg \
    --viz_image /path/to/IMG_0298.JPG \
    --out_dir ./out
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHT = ROOT / "weights" / "da_job01_best-epoch322-loss0.1250.pth"

from predict_image import (  # noqa: E402
    _save_mask_overlay,
    load_model,
    predict_one_strip512,
)


def main():
    ap = argparse.ArgumentParser(description="RootExp DA strip512→224 inference (one image).")
    ap.add_argument("--da_image", type=str, required=True, help="Preprocessed DA RGB (model input)")
    ap.add_argument(
        "--viz_image",
        type=str,
        default="",
        help="Original RGB for overlay; default = --da_image",
    )
    ap.add_argument("--weight", type=str, default=str(DEFAULT_WEIGHT))
    ap.add_argument("--out_dir", type=str, default=str(ROOT / "pred_outputs"))
    ap.add_argument("--strip_width", type=int, default=512)
    ap.add_argument("--strip_stride", type=int, default=512)
    ap.add_argument("--input_h", type=int, default=224)
    ap.add_argument("--input_w", type=int, default=224)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--drop_path_rate", type=float, default=0.05)
    ap.add_argument("--mlp_ratio", type=float, default=2.0)
    ap.add_argument("--hvss_d_state", type=int, default=8)
    ap.add_argument("--strip_blend", type=str, default="mean", choices=["mean", "hann"])
    args = ap.parse_args()

    import torch

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    da_img = Image.open(args.da_image).convert("RGB")
    viz_path = args.viz_image or args.da_image
    viz_img = Image.open(viz_path).convert("RGB")
    if da_img.size != viz_img.size:
        raise SystemExit(f"size mismatch da={da_img.size} viz={viz_img.size}")

    model = load_model(
        args.weight,
        device,
        strict=False,
        overrides={
            "drop_path_rate": args.drop_path_rate,
            "mlp_ratio": args.mlp_ratio,
            "hvss_d_state": args.hvss_d_state,
        },
    )
    pred = predict_one_strip512(
        model,
        da_img,
        device,
        strip_width=args.strip_width,
        strip_stride=args.strip_stride,
        train_in_hw=(args.input_h, args.input_w),
        global_mean=None,
        global_std=None,
        no_preprocess=False,
        strip_blend=args.strip_blend,
    )

    stem = Path(args.da_image).stem
    mask_path = os.path.join(args.out_dir, f"{stem}_mask.png")
    overlay_path = os.path.join(args.out_dir, f"{stem}_overlay.png")
    _save_mask_overlay(viz_img, pred, mask_path, overlay_path, args.threshold)
    print(f"[ok] mask  -> {mask_path}")
    print(f"[ok] blend -> {overlay_path}")


if __name__ == "__main__":
    main()
