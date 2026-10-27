"""
Same inference modes as predict_image.py (resize / strip / strip_fusion / tile), but loads
UNet++ (BatchNorm checkpoint) from duibi/da/unet++.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
_UNETPP_ROOT = _ROOT / "duibi" / "da" / "unet++"
sys.path.insert(0, str(_UNETPP_ROOT))

from models.unetplusplus_bn import UNetPlusPlus  # noqa: E402

from predict_image import (  # noqa: E402
    _collect_images,
    _parse_strip_fusion,
    load_train_stats,
    predict_one,
    predict_one_from_npy_slice,
)


def load_unetpp(weight_path, device, strict=True):
    model = UNetPlusPlus(num_classes=1, input_channels=3, deep_supervision=False).to(device)
    ckpt = torch.load(weight_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    return model


def parse_args():
    p = argparse.ArgumentParser(description="UNet++ (duibi/da) prediction — same CLI geometry as predict_image.py")
    p.add_argument("--image", type=str, default="", help="Single image path")
    p.add_argument("--input_dir", type=str, default="", help="Run on all images under this folder")
    p.add_argument("--data_npy", type=str, default="", help="NxHxWx3 npy slices")
    p.add_argument("--npy_max", type=int, default=5)
    p.add_argument("--npy_start", type=int, default=0)
    p.add_argument(
        "--weight",
        type=str,
        default=str(_UNETPP_ROOT / "best-epoch757-loss0.2393.pth"),
        help="UNet++ BN checkpoint",
    )
    p.add_argument("--out_dir", type=str, default="./pred_outputs_unetpp")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--input_h", type=int, default=224)
    p.add_argument("--input_w", type=int, default=224)
    p.add_argument(
        "--mode",
        type=str,
        default="resize",
        choices=["resize", "tile", "strip", "strip_fusion"],
    )
    p.add_argument("--strip_fusion", type=str, default="512:512,224:224")
    p.add_argument("--strip_width", type=int, default=512)
    p.add_argument("--strip_stride", type=int, default=512)
    p.add_argument("--strip_blend", type=str, default="mean", choices=["mean", "hann"])
    p.add_argument("--tile_h", type=int, default=128)
    p.add_argument("--tile_w", type=int, default=128)
    p.add_argument("--stride_h", type=int, default=96)
    p.add_argument("--stride_w", type=int, default=96)
    p.add_argument("--train_npy", type=str, default="")
    p.add_argument("--no_preprocess", action="store_true")
    p.add_argument("--strict", action="store_true", help="strict load_state_dict (default False)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats_path = "" if args.no_preprocess else (args.train_npy or (args.data_npy if args.data_npy else ""))
    global_mean = global_std = None
    if stats_path:
        global_mean, global_std = load_train_stats(stats_path)
        print(f"[train_stats] from={stats_path} mean={global_mean:.6f}, std={global_std:.6f}")
    elif args.no_preprocess:
        print("[input] no_preprocess")

    model = load_unetpp(args.weight, device, strict=args.strict)
    print(f"[UNet++] loaded {args.weight}")

    if args.data_npy:
        arr = np.load(args.data_npy, mmap_mode="r")
        n_total = arr.shape[0]
        i0 = max(0, args.npy_start)
        i1 = min(i0 + args.npy_max, n_total)
        if i0 >= n_total:
            raise SystemExit(f"npy_start {i0} >= N {n_total}")
        print(f"[data_npy] shape={arr.shape} [{i0}, {i1})")
        for idx in range(i0, i1):
            predict_one_from_npy_slice(
                model,
                np.asarray(arr[idx]),
                os.path.join(args.out_dir, f"npy_{idx:05d}_mask.png"),
                os.path.join(args.out_dir, f"npy_{idx:05d}_overlay.png"),
                device,
                size_hw=(args.input_h, args.input_w),
                threshold=args.threshold,
                global_mean=global_mean,
                global_std=global_std,
                no_preprocess=args.no_preprocess,
            )
            print(f"[ok] npy {idx}")
        print(f"Done. {i1 - i0} slice(s).")
        return

    paths = _collect_images(args)
    if not paths:
        raise SystemExit("Provide --data_npy, --image, or --input_dir")

    fusion_specs = _parse_strip_fusion(args.strip_fusion) if args.mode == "strip_fusion" else None
    if args.mode == "strip_fusion":
        print(f"[strip_fusion] specs={fusion_specs} train_in={args.input_h}x{args.input_w} strip_blend={args.strip_blend}")
    elif args.mode == "strip":
        print(f"[strip] strip_blend={args.strip_blend}")

    for image_path in paths:
        name = os.path.splitext(os.path.basename(image_path))[0]
        predict_one(
            model,
            image_path,
            os.path.join(args.out_dir, f"{name}_mask.png"),
            os.path.join(args.out_dir, f"{name}_overlay.png"),
            device,
            size_hw=(args.input_h, args.input_w),
            threshold=args.threshold,
            mode=args.mode,
            tile_hw=(args.tile_h, args.tile_w),
            stride_hw=(args.stride_h, args.stride_w),
            strip_width=args.strip_width,
            strip_stride=args.strip_stride,
            strip_specs=fusion_specs,
            strip_blend=args.strip_blend,
            global_mean=global_mean,
            global_std=global_std,
            no_preprocess=args.no_preprocess,
        )
        print(f"[ok] {image_path}")
    print(f"Done. {len(paths)} image(s).")


if __name__ == "__main__":
    main()
