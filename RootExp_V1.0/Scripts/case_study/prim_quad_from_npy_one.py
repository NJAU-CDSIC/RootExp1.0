#!/usr/bin/env python3
"""
Quad from prim npy: images | GT | Ours (RootBin) | UNet++ (duibi/prim).
Titles: images, groundtruth, our's, other's best.
Use --count N with --out_dir to export multiple indices starting at --idx (default 0).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
_REPO_PRIM = ROOT.parent.parent / "dataset" / "prim"
_UNETPP_PRIM = ROOT / "duibi" / "prim" / "unet++"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

from predict_image import load_model, load_train_stats, normalize_like_training

sys.path.insert(0, str(_UNETPP_PRIM))
from models.unetplusplus import UNetPlusPlus  # noqa: E402


def _mask_to_bw_uint8(m):
    if m.ndim == 3:
        m = m.squeeze(-1)
    if m.dtype != np.float64 and m.dtype != np.float32:
        fg = m > 0
    else:
        mx = float(np.max(m)) if m.size else 0.0
        fg = m > 0.5 if mx <= 1.0 else m > 0
    return (fg.astype(np.uint8) * 255)


def _prepare_input(img_np, global_mean, global_std, size_hw, device):
    ori_np_display = np.clip(img_np.astype(np.float64), 0.0, 255.0).astype(np.uint8)
    ori_h, ori_w = ori_np_display.shape[:2]
    ori_img = Image.fromarray(ori_np_display, mode="RGB")
    img_work = img_np.astype(np.float32)
    if (ori_h, ori_w) != (size_hw[0], size_hw[1]):
        pil_rs = ori_img.resize((size_hw[1], size_hw[0]), Image.BILINEAR)
        img_work = np.array(pil_rs, dtype=np.float32)
    img_norm = normalize_like_training(img_work, global_mean=global_mean, global_std=global_std)
    x = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return ori_np_display, x, ori_h, ori_w


def _pred_rootbin(model, x, ori_h, ori_w):
    with torch.no_grad():
        pred = model(x)
        pred = pred if not isinstance(pred, tuple) else pred[0]
        pred = F.interpolate(pred, size=(ori_h, ori_w), mode="bilinear", align_corners=False)
        return np.clip(pred.squeeze().detach().cpu().numpy(), 0.0, 1.0)


def _pred_unetpp(model, x, ori_h, ori_w):
    with torch.no_grad():
        pred = model(x)
    p = pred.squeeze().detach().cpu().numpy()
    if p.shape != (ori_h, ori_w):
        t = torch.from_numpy(p).unsqueeze(0).unsqueeze(0).float().to(x.device)
        t = F.interpolate(t, size=(ori_h, ori_w), mode="bilinear", align_corners=False)
        p = t.squeeze().cpu().numpy()
    return np.clip(p, 0.0, 1.0)


def pred_binary_fixed(prob: np.ndarray, threshold: float) -> np.ndarray:
    return ((prob >= threshold).astype(np.uint8)) * 255


def _try_font(size: int):
    for fp in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if os.path.isfile(fp):
            try:
                return ImageFont.truetype(fp, size)
            except OSError:
                pass
    return ImageFont.load_default()


def build_quad_rgba(ori_rgb_u8, gt_bw_u8, ours_u8, other_u8, titles, gap_px: int, title_h: int):
    h, w = ori_rgb_u8.shape[:2]
    total_w = w * 4 + gap_px * 3
    total_h = title_h + h
    canvas = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font = _try_font(max(14, title_h // 2))
    xs = [0, w + gap_px, 2 * (w + gap_px), 3 * (w + gap_px)]
    panels = [
        Image.fromarray(ori_rgb_u8, mode="RGB").convert("RGBA"),
        Image.fromarray(gt_bw_u8, mode="L").convert("RGBA"),
        Image.fromarray(ours_u8, mode="L").convert("RGBA"),
        Image.fromarray(other_u8, mode="L").convert("RGBA"),
    ]
    for x0, panel in zip(xs, panels):
        canvas.paste(panel, (x0, title_h), panel)
    for x0, title in zip(xs, titles):
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        tx = x0 + (w - tw) // 2
        ty = max(0, (title_h - (bbox[3] - bbox[1])) // 2)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((tx + dx, ty + dy), title, font=font, fill=(255, 255, 255, 255))
        draw.text((tx, ty), title, font=font, fill=(0, 0, 0, 255))
    return canvas


TITLES_DEFAULT = ["images", "groundtruth", "our's", "other's best"]


def run_one_index(
    idx,
    data,
    masks,
    model_ours,
    unet,
    gm,
    gs,
    size_hw,
    device,
    th_ours,
    th_unetpp,
    gap,
    title_h,
    titles,
):
    img_np = np.asarray(data[idx])
    m_np = np.asarray(masks[idx])
    ori_u8, x, oh, ow = _prepare_input(img_np, gm, gs, size_hw, device)
    prob_o = _pred_rootbin(model_ours, x, oh, ow)
    prob_u = _pred_unetpp(unet, x, oh, ow)
    gt_bw = _mask_to_bw_uint8(m_np)
    if gt_bw.shape != (oh, ow):
        gt_bw = np.array(Image.fromarray(gt_bw).resize((ow, oh), Image.NEAREST))
    ours_m = pred_binary_fixed(prob_o, th_ours)
    unet_m = pred_binary_fixed(prob_u, th_unetpp)
    return build_quad_rgba(ori_u8, gt_bw, ours_m, unet_m, titles, gap, title_h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_npy", type=str, default=str(_REPO_PRIM / "data_train.npy"))
    ap.add_argument("--mask_npy", type=str, default=str(_REPO_PRIM / "mask_train.npy"))
    ap.add_argument(
        "--weight_ours",
        type=str,
        default=str(
            ROOT
            / "results/RootBin_Block_Mamba_EAM_ASPP_ISIC2017_Tuesday_21_April_2026_02h_57m_46s/checkpoints/best.pth"
        ),
    )
    ap.add_argument(
        "--weight_unetpp",
        type=str,
        default=str(
            _UNETPP_PRIM
            / "results/UNetPlusPlus_ISIC2017_Tuesday_14_April_2026_22h_11m_18s/checkpoints/best-epoch645-loss0.2813.pth"
        ),
    )
    ap.add_argument("--train_npy", type=str, default=str(_REPO_PRIM / "data_train.npy"))
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--count", type=int, default=1, help="Export this many samples from idx into --out_dir")
    ap.add_argument("--th_ours", type=float, default=0.5, help="Binarize RootBin probs (try 0.05 if mask empty)")
    ap.add_argument("--th_unetpp", type=float, default=0.5, help="Binarize UNet++ probs")
    ap.add_argument("--gap", type=int, default=8)
    ap.add_argument("--title_h", type=int, default=40)
    ap.add_argument("--input_h", type=int, default=224)
    ap.add_argument("--input_w", type=int, default=224)
    ap.add_argument("--drop_path_rate", type=float, default=None)
    ap.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "prim_quad_preview" / "quad_train_idx00000.png"),
    )
    ap.add_argument("--out_dir", type=str, default=str(ROOT / "prim_quad_preview"))
    ap.add_argument(
        "--all_splits",
        action="store_true",
        help="Export all samples for prim data_train / data_val / data_test npy",
    )
    ap.add_argument("--npy_start", type=int, default=0)
    ap.add_argument("--npy_max", type=int, default=-1, help="-1 = all indices in split")
    ap.add_argument("--print_every", type=int, default=50)
    args = ap.parse_args()

    gm, gs = load_train_stats(args.train_npy)
    print(f"[stats] mean={gm:.6f} std={gs:.6f}")
    print(f"[ours] {args.weight_ours}")
    print(f"[unet++] {args.weight_unetpp}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overrides = {"drop_path_rate": args.drop_path_rate}
    model_ours = load_model(args.weight_ours, device, strict=False, overrides=overrides)

    unet = UNetPlusPlus(num_classes=1, input_channels=3, deep_supervision=False).to(device)
    ck = torch.load(args.weight_unetpp, map_location=device)
    unet.load_state_dict(ck, strict=True)
    unet.eval()

    size_hw = (args.input_h, args.input_w)
    titles = list(TITLES_DEFAULT)

    def loop_split(split_name, data_path, mask_path, out_root):
        if not os.path.isfile(data_path) or not os.path.isfile(mask_path):
            print(f"[skip] {split_name}: missing file")
            return 0
        data = np.load(data_path, mmap_mode="r")
        masks = np.load(mask_path, mmap_mode="r")
        n = data.shape[0]
        if masks.shape[0] != n:
            print(f"[skip] {split_name}: count mismatch")
            return 0
        subdir = os.path.join(out_root, split_name)
        os.makedirs(subdir, exist_ok=True)
        i0 = max(0, args.npy_start)
        i1 = n if args.npy_max < 0 else min(n, i0 + args.npy_max)
        done = 0
        for idx in range(i0, i1):
            quad = run_one_index(
                idx,
                data,
                masks,
                model_ours,
                unet,
                gm,
                gs,
                size_hw,
                device,
                args.th_ours,
                args.th_unetpp,
                args.gap,
                args.title_h,
                titles,
            )
            out_path = os.path.join(subdir, f"quad_{split_name}_idx_{idx:05d}.png")
            quad.save(out_path)
            done += 1
            if done % args.print_every == 0 or idx == i1 - 1:
                print(f"[{split_name}] {done}/{i1 - i0} (idx {idx})")
        print(f"[done] {split_name}: {done} -> {subdir}")
        return done

    if args.all_splits:
        pairs = [
            ("train", str(_REPO_PRIM / "data_train.npy"), str(_REPO_PRIM / "mask_train.npy")),
            ("val", str(_REPO_PRIM / "data_val.npy"), str(_REPO_PRIM / "mask_val.npy")),
            ("test", str(_REPO_PRIM / "data_test.npy"), str(_REPO_PRIM / "mask_test.npy")),
        ]
        total = 0
        for split_name, dp, mp in pairs:
            total += loop_split(split_name, dp, mp, args.out_dir)
        print(f"[all_splits] total {total} -> {args.out_dir}")
        return

    data = np.load(args.data_npy, mmap_mode="r")
    masks = np.load(args.mask_npy, mmap_mode="r")
    if masks.shape[0] != data.shape[0]:
        raise SystemExit("data/mask count mismatch")
    n = data.shape[0]

    if args.count > 1:
        i0 = max(0, args.idx)
        i1 = min(n, i0 + args.count)
        os.makedirs(args.out_dir, exist_ok=True)
        tag = Path(args.data_npy).stem.replace("data_", "")
        for idx in range(i0, i1):
            quad = run_one_index(
                idx,
                data,
                masks,
                model_ours,
                unet,
                gm,
                gs,
                size_hw,
                device,
                args.th_ours,
                args.th_unetpp,
                args.gap,
                args.title_h,
                titles,
            )
            out_path = os.path.join(args.out_dir, f"quad_{tag}_idx_{idx:05d}.png")
            quad.save(out_path)
            print(f"[saved] {out_path}")
        print(f"Done. {i1 - i0} quad(s) -> {args.out_dir}")
        return

    assert args.idx < data.shape[0]
    quad = run_one_index(
        args.idx,
        data,
        masks,
        model_ours,
        unet,
        gm,
        gs,
        size_hw,
        device,
        args.th_ours,
        args.th_unetpp,
        args.gap,
        args.title_h,
        titles,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    quad.save(args.out)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
