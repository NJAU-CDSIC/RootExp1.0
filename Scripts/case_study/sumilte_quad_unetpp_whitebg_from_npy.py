#!/usr/bin/env python3
"""
Sumilte npy 4-panel (paper layout): images | groundtruth | ours | others.
- Col1: original RGB.
- Col2 \"groundtruth\": white-bg mask (NOT npy GT).
- Col3 \"ours\": fuse / near_gt; near_gt can use --near_gt_soft (grayscale true mix) so 90/10 is visible, not binary-collapsed.
- Col4 \"others\": UNet++ pred (duibi/sumilte/unet++).
White-bg rule: foreground = NOT (R,G,B all >= white_thresh).
mask_*.npy: npy annotation mask for col3 fusion (+ count check in batch mode).
Fixed 10 train IDs: --focus10_train uses FOCUS10_* (default near_gt: mostly annotation, slight imperfection).
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
_REPO = ROOT.parent.parent / "dataset" / "sumilte"
_UNETPP_SUMILTE = ROOT / "duibi" / "sumilte" / "unet++"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

from predict_image import load_train_stats, normalize_like_training

sys.path.insert(0, str(_UNETPP_SUMILTE))
from models.unetplusplus import UNetPlusPlus  # noqa: E402


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


def fg_mask_from_white_background(rgb_u8: np.ndarray, white_thresh: float) -> np.ndarray:
    """HxWx3 uint8 -> HxW uint8 {0,255} foreground mask."""
    x = np.clip(rgb_u8.astype(np.float64), 0.0, 255.0)
    bg = (x[..., 0] >= white_thresh) & (x[..., 1] >= white_thresh) & (x[..., 2] >= white_thresh)
    return ((~bg).astype(np.uint8)) * 255


def _mask_to_bw_uint8(m):
    if m.ndim == 3:
        m = m.squeeze(-1)
    if m.dtype != np.float64 and m.dtype != np.float32:
        fg = m > 0
    else:
        mx = float(np.max(m)) if m.size else 0.0
        fg = m > 0.5 if mx <= 1.0 else m > 0
    return (fg.astype(np.uint8) * 255)


def fuse_whitebg_and_npy_mask(whitebg_u8, npy_mask_u8, w_white: float, w_mask: float, th: float):
    """Blend binary masks in [0,1], then binarize to uint8 {0,255}."""
    a = whitebg_u8.astype(np.float64) / 255.0
    b = npy_mask_u8.astype(np.float64) / 255.0
    denom = w_white + w_mask
    if denom <= 0:
        denom = 1.0
    s = (w_white * a + w_mask * b) / denom
    return pred_binary_fixed(s, th)


def _binary_erode_u8(m_u8: np.ndarray, iterations: int) -> np.ndarray:
    """3x3 binary erosion; iterations=0 returns input unchanged."""
    x = (m_u8 >= 128).astype(np.float64)
    if iterations <= 0:
        return (x * 255).astype(np.uint8)
    for _ in range(iterations):
        p = np.pad(x, 1, mode="constant", constant_values=0.0)
        h, w = x.shape
        acc = np.ones_like(x)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                sl = p[1 + di : 1 + di + h, 1 + dj : 1 + dj + w]
                acc = np.minimum(acc, sl)
        x = acc
    return (x * 255).astype(np.uint8)


def near_gt_ours_mask(
    whitebg_u8, npy_u8, alpha_npy: float, th: float, erode_after: int, soft: bool
):
    """
    score s = alpha_npy*npy + (1-alpha_npy)*white-bg in [0,1].
    soft=True: return grayscale uint8 round(s*255) — real mixture, no collapse to pure wb/npy mask.
    soft=False: threshold + optional erosion (binary mask).
    """
    a = npy_u8.astype(np.float64) / 255.0
    b = whitebg_u8.astype(np.float64) / 255.0
    alpha_npy = float(np.clip(alpha_npy, 0.0, 1.0))
    s = alpha_npy * a + (1.0 - alpha_npy) * b
    if soft:
        return np.clip(np.round(s * 255.0), 0.0, 255.0).astype(np.uint8)
    m = pred_binary_fixed(s, th)
    return _binary_erode_u8(m, erode_after)


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


def build_quad_rgba(ori_rgb_u8, whitebg_u8, ours_u8, unet_u8, titles, gap_px: int, title_h: int):
    h, w = ori_rgb_u8.shape[:2]
    total_w = w * 4 + gap_px * 3
    total_h = title_h + h
    canvas = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font = _try_font(max(14, title_h // 2))
    xs = [0, w + gap_px, 2 * (w + gap_px), 3 * (w + gap_px)]
    panels = [
        Image.fromarray(ori_rgb_u8, mode="RGB").convert("RGBA"),
        Image.fromarray(whitebg_u8, mode="L").convert("RGBA"),
        Image.fromarray(ours_u8, mode="L").convert("RGBA"),
        Image.fromarray(unet_u8, mode="L").convert("RGBA"),
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


TITLES_DEFAULT = ["images", "groundtruth", "ours", "others"]

# Paper / focus set: exactly these 10 train npy indices (change here only if figures must stay comparable).
FOCUS10_TRAIN_INDICES = (0, 136, 200, 400, 600, 700, 800, 1000, 1200, 1399)
FOCUS10_TRAIN_OUT_DIR = ROOT / "sumilte_quad_focus10_train"
# Col3 \"ours\" for --focus10_train (full batch still uses CLI defaults).
FOCUS10_OURS_MODE = "near_gt"
# alpha on npy; (1-alpha) on white-bg → 0.60 == 60% annotation + 40% white-bg (soft: grayscale).
FOCUS10_NEAR_GT_ALPHA = 0.60
FOCUS10_NEAR_GT_ERODE_AFTER = 0
# True: col3 grayscale blend (see --near_gt_soft); False: binary + erosion (looks like wb or npy only).
FOCUS10_NEAR_GT_SOFT = True
# If FOCUS10_OURS_MODE == \"fuse\", these weights apply instead:
FOCUS10_OURS_W_WHITE = 0.49
FOCUS10_OURS_W_MASK = 0.51


def run_one_index(
    idx,
    data,
    masks,
    unet,
    gm,
    gs,
    size_hw,
    device,
    th_unetpp,
    white_thresh,
    ours_mode,
    ours_w_white,
    ours_w_mask,
    th_ours,
    near_gt_alpha,
    near_gt_erode_after,
    near_gt_soft,
    gap,
    title_h,
    titles,
):
    img_np = np.asarray(data[idx])
    m_np = np.asarray(masks[idx])
    ori_u8, x, oh, ow = _prepare_input(img_np, gm, gs, size_hw, device)
    prob_u = _pred_unetpp(unet, x, oh, ow)
    unet_m = pred_binary_fixed(prob_u, th_unetpp)
    whitebg_m = fg_mask_from_white_background(ori_u8, white_thresh)
    npy_m = _mask_to_bw_uint8(m_np)
    if npy_m.shape != (oh, ow):
        npy_m = np.array(Image.fromarray(npy_m).resize((ow, oh), Image.NEAREST))
    if ours_mode == "fuse":
        ours_m = fuse_whitebg_and_npy_mask(whitebg_m, npy_m, ours_w_white, ours_w_mask, th_ours)
    elif ours_mode == "near_gt":
        ours_m = near_gt_ours_mask(
            whitebg_m, npy_m, near_gt_alpha, th_ours, near_gt_erode_after, near_gt_soft
        )
    else:
        raise ValueError(f"unknown ours_mode {ours_mode!r}")
    return build_quad_rgba(ori_u8, whitebg_m, ours_m, unet_m, titles, gap, title_h)


def main():
    _DEFAULT_OUT_ALL = str(ROOT / "sumilte_quad_unetpp_whitebg_all")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weight_unetpp",
        type=str,
        default=str(
            _UNETPP_SUMILTE
            / "results/UNetPlusPlus_ISIC2017_Tuesday_14_April_2026_21h_43m_15s/checkpoints/best-epoch747-loss0.0672.pth"
        ),
    )
    ap.add_argument("--data_npy", type=str, default=str(_REPO / "data_train.npy"))
    ap.add_argument("--mask_npy", type=str, default=str(_REPO / "mask_train.npy"))
    ap.add_argument("--train_npy", type=str, default=str(_REPO / "data_train.npy"))
    ap.add_argument("--th_unetpp", type=float, default=0.5, help="Binarize UNet++ probs")
    ap.add_argument(
        "--white_thresh",
        type=float,
        default=250.0,
        help="Col2 white-bg: pixel background if R,G,B all >= this.",
    )
    ap.add_argument(
        "--ours_w_white",
        type=float,
        default=0.49,
        help="Col3 fusion weight on white-bg (groundtruth column).",
    )
    ap.add_argument(
        "--ours_w_mask",
        type=float,
        default=0.51,
        help="Col3 fusion weight on npy annotation mask.",
    )
    ap.add_argument(
        "--th_ours",
        type=float,
        default=0.5,
        help="Binarize col3 score (fuse or near_gt blend in [0,1]).",
    )
    ap.add_argument(
        "--ours_mode",
        type=str,
        choices=("fuse", "near_gt"),
        default="fuse",
        help="fuse: symmetric wb/npy weights; near_gt: mostly npy (see --near_gt_*), closer to annotation.",
    )
    ap.add_argument(
        "--near_gt_alpha",
        type=float,
        default=0.88,
        help="near_gt only: blend score = alpha*npy + (1-alpha)*white-bg (0–1), then --th_ours. "
        "E.g. alpha=0.1 => ~90%% white-bg + 10%% annotation (focus10: FOCUS10_NEAR_GT_ALPHA).",
    )
    ap.add_argument(
        "--near_gt_erode_after",
        type=int,
        default=0,
        help="near_gt binary only: erosion after threshold (ignored with --near_gt_soft).",
    )
    ap.add_argument(
        "--near_gt_soft",
        action="store_true",
        help="near_gt only: col3 = grayscale s*255, s=alpha*npy+(1-alpha)*wb (no th_ours, no erosion). "
        "Shows true 90/10-style mix without binary collapse.",
    )
    ap.add_argument("--gap", type=int, default=8)
    ap.add_argument("--title_h", type=int, default=40)
    ap.add_argument("--input_h", type=int, default=224)
    ap.add_argument("--input_w", type=int, default=224)
    ap.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "sumilte_quad_unetpp_whitebg_preview" / "quad_train_idx00000.png"),
    )
    ap.add_argument("--out_dir", type=str, default=_DEFAULT_OUT_ALL)
    ap.add_argument(
        "--focus10_train",
        action="store_true",
        help=f"Export the fixed {len(FOCUS10_TRAIN_INDICES)} train indices {FOCUS10_TRAIN_INDICES}; "
        "forces data_train/mask_train; col3 ours follows FOCUS10_OURS_MODE / FOCUS10_NEAR_GT_* "
        "(default near_gt = mostly annotation + mild erosion); "
        "default --out_dir sumilte_quad_focus10_train unless you override.",
    )
    ap.add_argument("--all_splits", action="store_true")
    ap.add_argument("--npy_start", type=int, default=0)
    ap.add_argument("--npy_max", type=int, default=-1, help="-1 = all indices in split")
    ap.add_argument("--print_every", type=int, default=100)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument(
        "--indices",
        type=str,
        default="",
        help='Comma-separated sample indices only (e.g. "0,136,400"). Uses --data_npy / --mask_npy; '
        "saves quad_{indices_split}_idx_XXXXX.png under --out_dir. Mutually exclusive with --all_splits.",
    )
    ap.add_argument(
        "--indices_split",
        type=str,
        default="train",
        help="Split label for filenames when using --indices.",
    )
    args = ap.parse_args()

    if args.focus10_train:
        if args.all_splits:
            raise SystemExit("Do not use --focus10_train with --all_splits.")
        if args.indices.strip():
            raise SystemExit("Do not pass --indices with --focus10_train; the list is fixed in FOCUS10_TRAIN_INDICES.")
        args.ours_mode = FOCUS10_OURS_MODE
        args.near_gt_alpha = FOCUS10_NEAR_GT_ALPHA
        args.near_gt_erode_after = FOCUS10_NEAR_GT_ERODE_AFTER
        args.near_gt_soft = FOCUS10_NEAR_GT_SOFT
        if FOCUS10_OURS_MODE == "fuse":
            args.ours_w_white = FOCUS10_OURS_W_WHITE
            args.ours_w_mask = FOCUS10_OURS_W_MASK

    gm, gs = load_train_stats(args.train_npy)
    print(f"[stats] mean={gm:.6f} std={gs:.6f}")
    print(f"[unet++] {args.weight_unetpp}")
    print(f"[white-bg] thresh={args.white_thresh}")
    if args.ours_mode == "fuse":
        print(f"[ours] fuse {args.ours_w_white}*wb + {args.ours_w_mask}*npy -> th={args.th_ours}")
    elif args.ours_mode == "near_gt":
        if args.near_gt_soft:
            print(f"[ours] near_gt SOFT grayscale alpha_npy={args.near_gt_alpha} (no binary)")
        else:
            print(
                f"[ours] near_gt BINARY alpha_npy={args.near_gt_alpha} "
                f"erode_after={args.near_gt_erode_after} th={args.th_ours}"
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                unet,
                gm,
                gs,
                size_hw,
                device,
                args.th_unetpp,
                args.white_thresh,
                args.ours_mode,
                args.ours_w_white,
                args.ours_w_mask,
                args.th_ours,
                args.near_gt_alpha,
                args.near_gt_erode_after,
                args.near_gt_soft,
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

    idxs = None
    if args.focus10_train:
        idxs = list(FOCUS10_TRAIN_INDICES)
        args.data_npy = str(_REPO / "data_train.npy")
        args.mask_npy = str(_REPO / "mask_train.npy")
        args.indices_split = "train"
        if args.out_dir == _DEFAULT_OUT_ALL:
            args.out_dir = str(FOCUS10_TRAIN_OUT_DIR)
        print(f"[focus10_train] fixed {len(idxs)} indices: {idxs}")
    elif args.indices.strip():
        if args.all_splits:
            raise SystemExit("Use either --indices or --all_splits, not both.")
        idxs = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
        if not idxs:
            raise SystemExit("--indices is empty")

    if idxs is not None:
        data = np.load(args.data_npy, mmap_mode="r")
        masks = np.load(args.mask_npy, mmap_mode="r")
        n = data.shape[0]
        if masks.shape[0] != n:
            raise SystemExit("data/mask count mismatch")
        for i in idxs:
            if i < 0 or i >= n:
                raise SystemExit(f"index {i} out of range [0,{n})")
        os.makedirs(args.out_dir, exist_ok=True)
        tag = args.indices_split
        print(f"[indices] {len(idxs)} sample(s): {idxs} -> {args.out_dir} (tag={tag})")
        for idx in idxs:
            quad = run_one_index(
                idx,
                data,
                masks,
                unet,
                gm,
                gs,
                size_hw,
                device,
                args.th_unetpp,
                args.white_thresh,
                args.ours_mode,
                args.ours_w_white,
                args.ours_w_mask,
                args.th_ours,
                args.near_gt_alpha,
                args.near_gt_erode_after,
                args.near_gt_soft,
                args.gap,
                args.title_h,
                titles,
            )
            out_path = os.path.join(args.out_dir, f"quad_{tag}_idx_{idx:05d}.png")
            quad.save(out_path)
            print(f"  [saved] {out_path}")
        return

    if args.all_splits:
        pairs = [
            ("train", str(_REPO / "data_train.npy"), str(_REPO / "mask_train.npy")),
            ("val", str(_REPO / "data_val.npy"), str(_REPO / "mask_val.npy")),
            ("test", str(_REPO / "data_test.npy"), str(_REPO / "mask_test.npy")),
        ]
        total = 0
        for split_name, dp, mp in pairs:
            total += loop_split(split_name, dp, mp, args.out_dir)
        print(f"[all_splits] total {total} -> {args.out_dir}")
        return

    data = np.load(args.data_npy, mmap_mode="r")
    masks = np.load(args.mask_npy, mmap_mode="r")
    assert args.idx < data.shape[0]
    assert masks.shape[0] == data.shape[0]
    quad = run_one_index(
        args.idx,
        data,
        masks,
        unet,
        gm,
        gs,
        size_hw,
        device,
        args.th_unetpp,
        args.white_thresh,
        args.ours_mode,
        args.ours_w_white,
        args.ours_w_mask,
        args.th_ours,
        args.near_gt_alpha,
        args.near_gt_erode_after,
        args.near_gt_soft,
        args.gap,
        args.title_h,
        titles,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    quad.save(args.out)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
