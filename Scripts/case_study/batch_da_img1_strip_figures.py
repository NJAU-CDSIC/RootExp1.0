"""
For every image in dataset/da/images: strip infer on DA (RootBin + UNet++), overlays on
matching img_1 original, save masks/overlays, then build titled six-panel vertical PNG.

Requires same WxH between dataset/da/images/<stem>.* and img_1/<stem>.* (case/extension).
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from compose_six_panel_rgba import DEFAULT_TITLES, compose_vertical_six  # noqa: E402
from demo_img1_strip_predict_one import predict_strip_unetpp  # noqa: E402
from predict_image import (  # noqa: E402
    _save_mask_overlay,
    load_model as load_rootbin,
    predict_one_strip512,
)

_UNETPP_ROOT = ROOT / "duibi" / "da" / "unet++"
sys.path.insert(0, str(_UNETPP_ROOT))
from models.unetplusplus_bn import UNetPlusPlus  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}


def resolve_img1(img1_dir: Path, stem: str) -> Optional[Path]:
    for ext in (".JPG", ".jpg", ".jpeg", ".JPEG", ".png", ".PNG"):
        p = img1_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--da_images", type=str, default="/home/young/桌面/testGeModel/dataset/da/images")
    ap.add_argument("--da_masks", type=str, default="/home/young/桌面/testGeModel/dataset/da/masks")
    ap.add_argument("--img1_dir", type=str, default="/home/young/桌面/rape_root_seg/dataset/img_1")
    ap.add_argument("--out_root", type=str, default=str(ROOT / "da_batch_img1_figures"))
    ap.add_argument(
        "--rootbin_weight",
        type=str,
        default="/home/young/下载/RootBin_Block_Mamba_EAM_ASPP/results/machine_split/job01_lr3e4_wd1e2_dp005_mlp20_ds8_bs8_20260422_154329/checkpoints/best-epoch322-loss0.1250.pth",
    )
    ap.add_argument("--unetpp_weight", type=str, default=str(_UNETPP_ROOT / "best-epoch757-loss0.2393.pth"))
    ap.add_argument("--strip_width", type=int, default=512)
    ap.add_argument("--strip_stride", type=int, default=512)
    ap.add_argument("--input_h", type=int, default=224)
    ap.add_argument("--input_w", type=int, default=224)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--train_npy", type=str, default="")
    ap.add_argument("--no_preprocess", action="store_true")
    ap.add_argument("--drop_path_rate", type=float, default=0.05)
    ap.add_argument("--mlp_ratio", type=float, default=2.0)
    ap.add_argument("--hvss_d_state", type=int, default=8)
    ap.add_argument("--panel_w", type=int, default=900)
    ap.add_argument("--gap", type=int, default=24)
    ap.add_argument("--title_h", type=int, default=48)
    ap.add_argument("--title_gap", type=int, default=10)
    ap.add_argument("--font_size", type=int, default=22)
    ap.add_argument("--font", type=str, default="")
    args = ap.parse_args()

    da_dir = Path(args.da_images)
    mask_dir = Path(args.da_masks)
    img1_dir = Path(args.img1_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_mean = global_std = None
    if args.train_npy and not args.no_preprocess:
        from predict_image import load_train_stats

        global_mean, global_std = load_train_stats(args.train_npy)
        print(f"[train_stats] mean={global_mean:.6f} std={global_std:.6f}")

    print("[load] RootBin …")
    rb = load_rootbin(
        args.rootbin_weight,
        device,
        strict=False,
        overrides={
            "drop_path_rate": args.drop_path_rate,
            "mlp_ratio": args.mlp_ratio,
            "hvss_d_state": args.hvss_d_state,
        },
    )
    print("[load] UNet++ …")
    unet = UNetPlusPlus(num_classes=1, input_channels=3, deep_supervision=False).to(device)
    ckpt = torch.load(args.unetpp_weight, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    unet.load_state_dict(state, strict=True)

    paths = sorted(
        p
        for p in da_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {e.lower() for e in IMG_EXTS}
    )

    ok = skip = 0
    for da_path in paths:
        stem = da_path.stem
        viz_path = resolve_img1(img1_dir, stem)
        gt_path = mask_dir / f"{stem}.png"
        if viz_path is None:
            print(f"[skip] no img_1 for stem={stem}")
            skip += 1
            continue
        if not gt_path.is_file():
            print(f"[skip] no mask {gt_path.name}")
            skip += 1
            continue

        da_img = Image.open(da_path).convert("RGB")
        viz_img = Image.open(viz_path).convert("RGB")
        if da_img.size != viz_img.size:
            print(f"[skip] size mismatch {stem}: da={da_img.size} img1={viz_img.size}")
            skip += 1
            continue

        gt_probe = Image.open(gt_path).convert("L")
        if gt_probe.size != da_img.size:
            print(f"[skip] gt size {gt_probe.size} != da {da_img.size} ({stem})")
            skip += 1
            continue

        sub = out_root / stem
        sub.mkdir(parents=True, exist_ok=True)

        pred_rb = predict_one_strip512(
            rb,
            da_img,
            device,
            strip_width=args.strip_width,
            strip_stride=args.strip_stride,
            train_in_hw=(args.input_h, args.input_w),
            global_mean=global_mean,
            global_std=global_std,
            no_preprocess=args.no_preprocess,
        )
        _save_mask_overlay(
            viz_img,
            pred_rb,
            str(sub / f"{stem}_rootbin_mask.png"),
            str(sub / f"{stem}_rootbin_overlay.png"),
            args.threshold,
        )

        pred_u = predict_strip_unetpp(
            unet,
            da_img,
            device,
            args.strip_width,
            args.strip_stride,
            (args.input_h, args.input_w),
            global_mean,
            global_std,
            args.no_preprocess,
        )
        _save_mask_overlay(
            viz_img,
            pred_u,
            str(sub / f"{stem}_unetpp_mask.png"),
            str(sub / f"{stem}_unetpp_overlay.png"),
            args.threshold,
        )

        gt_np = np.array(gt_probe, dtype=np.float32)
        if gt_np.max() > 1.5:
            gt_np = np.clip(gt_np / 255.0, 0.0, 1.0)
        else:
            gt_np = np.clip(gt_np, 0.0, 1.0)
        _save_mask_overlay(
            viz_img,
            gt_np,
            str(sub / f"{stem}_gt_mask.png"),
            str(sub / f"{stem}_gt_overlay.png"),
            0.5,
        )

        six_out = sub / f"{stem}_six_vertical_titled.png"
        compose_vertical_six(
            [
                str(viz_path),
                str(sub / f"{stem}_gt_mask.png"),
                str(sub / f"{stem}_rootbin_mask.png"),
                str(sub / f"{stem}_rootbin_overlay.png"),
                str(sub / f"{stem}_unetpp_mask.png"),
                str(sub / f"{stem}_unetpp_overlay.png"),
            ],
            str(six_out),
            titles=list(DEFAULT_TITLES),
            panel_w=args.panel_w,
            gap=args.gap,
            title_h=args.title_h,
            title_gap=args.title_gap,
            font_size=args.font_size,
            font_path=args.font,
        )
        print(f"[ok] {stem} → {six_out}")
        ok += 1

    print(f"Done. ok={ok} skip={skip} out_root={out_root}")


if __name__ == "__main__":
    main()
