"""
One-image demo: models MUST run on dataset/da/images (preprocessed RGB, matches training).
rape_root_seg/img_1 is the same scene without that preprocessing — use it only as the
background for overlay visualization (demo “跳过预处理”指的是给观众看原图叠掩码，不是说把 img_1 喂给网络).

Flow: strip512→224 inference on --da_image → masks/overlays drawn on --viz_image (img_1).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
_UNETPP_ROOT = ROOT / "duibi" / "da" / "unet++"
sys.path.insert(0, str(ROOT))
from predict_image import (  # noqa: E402
    _strip_x_coords,
    input_from_rgb_numpy,
    load_model as load_rootbin,
    predict_one_strip512,
    _save_mask_overlay,
)
sys.path.insert(0, str(_UNETPP_ROOT))
from models.unetplusplus_bn import UNetPlusPlus  # noqa: E402


def predict_strip_unetpp(model, ori_img, device, strip_width, strip_stride, train_in_hw, global_mean, global_std, no_prep):
    ori_w, ori_h = ori_img.size
    pred_sum = np.zeros((ori_h, ori_w), dtype=np.float32)
    pred_cnt = np.zeros((ori_h, ori_w), dtype=np.float32)
    th, tw = train_in_hw
    xs = _strip_x_coords(ori_w, strip_width, strip_stride)
    model.eval()
    for x0 in xs:
        x1 = min(x0 + strip_width, ori_w)
        strip_img = ori_img.crop((x0, 0, x1, ori_h))
        w_strip, h_strip = strip_img.size
        strip_rs = strip_img.resize((tw, th), Image.BILINEAR)
        img_np = input_from_rgb_numpy(
            np.array(strip_rs, dtype=np.float32),
            global_mean=global_mean,
            global_std=global_std,
            no_preprocess=no_prep,
        )
        x = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            pred = model(x)
            pred = F.interpolate(pred, size=(h_strip, w_strip), mode="bilinear", align_corners=False)
            p = pred.squeeze().detach().cpu().numpy()
        pred_sum[:, x0:x1] += p[:, :w_strip]
        pred_cnt[:, x0:x1] += 1.0
    pred_cnt[pred_cnt == 0] = 1.0
    return pred_sum / pred_cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--da_image",
        type=str,
        default="/home/young/桌面/testGeModel/dataset/da/images/1-1_fan_.jpg",
        help="Preprocessed DA image — this RGB is fed to RootBin + UNet++ (strip pipeline).",
    )
    ap.add_argument(
        "--viz_image",
        type=str,
        default="/home/young/桌面/rape_root_seg/dataset/img_1/1-1_fan_.JPG",
        help="Original img_1 for visualization only (overlays); WxH must match --da_image.",
    )
    ap.add_argument(
        "--gt_mask",
        type=str,
        default="/home/young/桌面/testGeModel/dataset/da/masks/1-1_fan_.png",
        help="GT mask PNG; empty string to skip GT overlay save",
    )
    ap.add_argument(
        "--rootbin_weight",
        type=str,
        default="/home/young/下载/RootBin_Block_Mamba_EAM_ASPP/results/machine_split/job01_lr3e4_wd1e2_dp005_mlp20_ds8_bs8_20260422_154329/checkpoints/best-epoch322-loss0.1250.pth",
    )
    ap.add_argument(
        "--unetpp_weight",
        type=str,
        default=str(ROOT / "duibi" / "da" / "unet++" / "best-epoch757-loss0.2393.pth"),
    )
    ap.add_argument("--out_dir", type=str, default=str(ROOT / "demo_img1_strip_one"))
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
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_mean = global_std = None
    if args.train_npy and not args.no_preprocess:
        from predict_image import load_train_stats

        global_mean, global_std = load_train_stats(args.train_npy)
        print(f"[train_stats] mean={global_mean:.6f} std={global_std:.6f}")

    da_img = Image.open(args.da_image).convert("RGB")
    viz_img = Image.open(args.viz_image).convert("RGB")
    if da_img.size != viz_img.size:
        raise SystemExit(
            f"--da_image size {da_img.size} != --viz_image {viz_img.size}; overlays must align."
        )
    stem = Path(args.da_image).stem

    print("[RootBin] strip infer on DA image …")
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
        os.path.join(args.out_dir, f"{stem}_rootbin_mask.png"),
        os.path.join(args.out_dir, f"{stem}_rootbin_overlay.png"),
        args.threshold,
    )
    print(f"[ok] RootBin (inp=DA) overlay(img_1) → {args.out_dir}/{stem}_rootbin_*")

    print("[UNet++ BN] strip infer on DA image …")
    unet = UNetPlusPlus(num_classes=1, input_channels=3, deep_supervision=False).to(device)
    ckpt = torch.load(args.unetpp_weight, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    unet.load_state_dict(state, strict=True)
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
        os.path.join(args.out_dir, f"{stem}_unetpp_mask.png"),
        os.path.join(args.out_dir, f"{stem}_unetpp_overlay.png"),
        args.threshold,
    )
    print(f"[ok] UNet++ (inp=DA) overlay(img_1) → {args.out_dir}/{stem}_unetpp_*")

    if args.gt_mask and os.path.isfile(args.gt_mask):
        gt = Image.open(args.gt_mask).convert("L")
        if gt.size != da_img.size:
            print(f"[skip GT overlay] mask size {gt.size} != da_image {da_img.size}")
        else:
            gt_np = np.array(gt, dtype=np.float32)
            if gt_np.max() > 1.5:
                gt_np = np.clip(gt_np / 255.0, 0.0, 1.0)
            else:
                gt_np = np.clip(gt_np, 0.0, 1.0)
            _save_mask_overlay(
                viz_img,
                gt_np,
                os.path.join(args.out_dir, f"{stem}_gt_mask.png"),
                os.path.join(args.out_dir, f"{stem}_gt_overlay.png"),
                0.5,
            )
            print(f"[ok] GT overlay → {stem}_gt_*")
    else:
        print("[info] no GT mask saved")

    print("Done.")


if __name__ == "__main__":
    main()
