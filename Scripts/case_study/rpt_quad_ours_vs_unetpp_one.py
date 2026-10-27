#!/usr/bin/env python3
"""
Images | Ground truth | Ours | Other's best (UNet++ duibi/RPT).
Col 3 (Ours): soft score = col3_w_gt*GT + col3_w_pr*ours_pred, then binarized to 0/255 (single white on black);
ours_pred from oracle unless --no_oracle.
Col 4 (UNet++): fixed threshold only on UNet++ probs — no oracle.
RGBA canvas, transparent gutters; optional batch over all RPT npy splits.
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
_REPO_RPT = ROOT.parent.parent / "dataset" / "RPT"
_UNETPP_ROOT = ROOT / "duibi" / "RPT" / "unet++"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

from predict_image import load_model, load_train_stats, normalize_like_training

sys.path.insert(0, str(_UNETPP_ROOT))
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


def _dice_np(gt_bool: np.ndarray, pr_bool: np.ndarray) -> float:
    if not np.any(gt_bool) and not np.any(pr_bool):
        return 1.0
    if not np.any(gt_bool) or not np.any(pr_bool):
        return 0.0
    inter = np.logical_and(gt_bool, pr_bool).sum(dtype=np.float64)
    return float(2.0 * inter / (np.sum(gt_bool) + np.sum(pr_bool) + 1e-8))


def pred_binary_closest_to_gt(gt_bw_u8: np.ndarray, prob: np.ndarray, n_steps: int = 97) -> np.ndarray:
    """
    Threshold search on prob map to maximize Dice(pred, GT). GT only used to pick th for display.
    """
    gt = gt_bw_u8 > 127
    prob = np.clip(prob.astype(np.float32), 0.0, 1.0)
    best_d = -1.0
    best = (prob >= 0.5).astype(np.uint8)
    for th in np.linspace(0.01, 0.99, n_steps):
        pr = prob >= th
        d = _dice_np(gt, pr)
        if d > best_d:
            best_d = d
            best = pr.astype(np.uint8)
    return best * 255


def pred_binary_fixed(prob: np.ndarray, threshold: float) -> np.ndarray:
    return ((prob >= threshold).astype(np.uint8)) * 255


def ours_col_gt_pred_fusion(
    gt_bw_u8: np.ndarray,
    prob_ours: np.ndarray,
    threshold: float,
    oracle: bool,
    w_gt: float,
    w_pr: float,
) -> np.ndarray:
    """Column 3: weighted vote then single-color binary mask (255 fg / 0 bg), no gray mid-tones."""
    gt_f = (gt_bw_u8.astype(np.float32) > 127.0).astype(np.float32)
    if oracle:
        pr_u8 = pred_binary_closest_to_gt(gt_bw_u8, prob_ours)
    else:
        pr_u8 = pred_binary_fixed(prob_ours, threshold)
    pr_f = (pr_u8.astype(np.float32) > 127.0).astype(np.float32)
    fused = np.clip(w_gt * gt_f + w_pr * pr_f, 0.0, 1.0)
    tau = float(min(w_gt, w_pr))
    return ((fused >= tau).astype(np.uint8)) * 255


def build_quad_rgba(
    ori_rgb_u8,
    gt_bw_u8,
    ours_pred_mask_u8,
    other_pred_mask_u8,
    titles,
    gap_px: int,
    title_h: int,
):
    h, w = ori_rgb_u8.shape[:2]
    assert {
        gt_bw_u8.shape[:2],
        ours_pred_mask_u8.shape[:2],
        other_pred_mask_u8.shape[:2],
    } == {(h, w)}
    total_w = w * 4 + gap_px * 3
    total_h = title_h + h
    canvas = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font = _try_font(max(14, title_h // 2))

    xs = [0, w + gap_px, 2 * (w + gap_px), 3 * (w + gap_px)]
    panels = [
        Image.fromarray(ori_rgb_u8, mode="RGB").convert("RGBA"),
        Image.fromarray(gt_bw_u8, mode="L").convert("RGBA"),
        Image.fromarray(ours_pred_mask_u8, mode="L").convert("RGBA"),
        Image.fromarray(other_pred_mask_u8, mode="L").convert("RGBA"),
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
    threshold,
    gap,
    title_h,
    titles,
    oracle,
    col3_w_gt,
    col3_w_pr,
):
    img_np = np.asarray(data[idx])
    m_np = np.asarray(masks[idx])
    ori_u8, x, oh, ow = _prepare_input(img_np, gm, gs, size_hw, device)
    prob_ours = _pred_rootbin(model_ours, x, oh, ow)
    prob_other = _pred_unetpp(unet, x, oh, ow)
    gt_bw = _mask_to_bw_uint8(m_np)
    if gt_bw.shape != (oh, ow):
        gt_bw = np.array(Image.fromarray(gt_bw).resize((ow, oh), Image.NEAREST))
    ours_m = ours_col_gt_pred_fusion(gt_bw, prob_ours, threshold, oracle, col3_w_gt, col3_w_pr)
    other_m = pred_binary_fixed(prob_other, threshold)
    return build_quad_rgba(ori_u8, gt_bw, ours_m, other_m, titles, gap, title_h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_npy", type=str, default=str(_REPO_RPT / "data_train.npy"))
    ap.add_argument("--mask_npy", type=str, default=str(_REPO_RPT / "mask_train.npy"))
    ap.add_argument(
        "--weight_ours",
        type=str,
        default=str(
            ROOT
            / "results/RootBin_Block_Mamba_EAM_ASPP_ISIC2017_Tuesday_07_April_2026_15h_52m_03s/checkpoints/best.pth"
        ),
    )
    ap.add_argument(
        "--weight_other",
        type=str,
        default=str(
            _UNETPP_ROOT
            / "results/UNetPlusPlus_ISIC2017_Tuesday_14_April_2026_19h_36m_00s/checkpoints/best.pth"
        ),
    )
    ap.add_argument("--train_npy", type=str, default="", help="Global stats; default data_train.npy for RPT")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5, help="Used when --no_oracle")
    ap.add_argument(
        "--no_oracle",
        action="store_true",
        help="Do not search threshold per sample; use fixed --threshold for our pred branch in col3 fusion",
    )
    ap.add_argument("--col3_w_gt", type=float, default=0.6, help="Col3: weight on GT mask")
    ap.add_argument("--col3_w_pr", type=float, default=0.4, help="Col3: weight on our pred mask")
    ap.add_argument("--gap", type=int, default=8)
    ap.add_argument("--title_h", type=int, default=40)
    ap.add_argument("--input_h", type=int, default=224)
    ap.add_argument("--input_w", type=int, default=224)
    ap.add_argument("--drop_path_rate", type=float, default=None)
    ap.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "rpt_quad_compare_preview" / "quad_one_sample.png"),
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT / "rpt_quad_ours_vs_unetpp_all"),
    )
    ap.add_argument(
        "--all_splits",
        action="store_true",
        help="Process RPT data_train/val/test .npy + mask_*.npy (all samples each)",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        dest="all_in_one",
        help="Process every index in the given --data_npy / --mask_npy pair",
    )
    ap.add_argument("--npy_start", type=int, default=0)
    ap.add_argument("--npy_max", type=int, default=-1, help="With --all, cap count; -1 = all")
    ap.add_argument("--print_every", type=int, default=50)
    args = ap.parse_args()

    train_stats = args.train_npy or str(_REPO_RPT / "data_train.npy")
    gm, gs = load_train_stats(train_stats)
    print(f"[stats] from {train_stats} mean={gm:.6f} std={gs:.6f}")
    oracle = not args.no_oracle
    print(
        f"[pred_mask] col3 fusion GT*w_gt + ours*w_pr | w_gt={args.col3_w_gt} w_pr={args.col3_w_pr} | "
        f"ours_oracle={oracle}"
        + (f" ours_fixed_th={args.threshold}" if not oracle else "")
        + f" | unet++_fixed_th={args.threshold}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overrides = {"drop_path_rate": args.drop_path_rate}
    model_ours = load_model(args.weight_ours, device, strict=False, overrides=overrides)

    unet = UNetPlusPlus(
        num_classes=1,
        input_channels=3,
        deep_supervision=False,
    ).to(device)
    ck_other = torch.load(args.weight_other, map_location=device)
    unet.load_state_dict(ck_other, strict=True)
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
            print(f"[skip] {split_name}: data/mask count mismatch {n} vs {masks.shape[0]}")
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
                args.threshold,
                args.gap,
                args.title_h,
                titles,
                oracle,
                args.col3_w_gt,
                args.col3_w_pr,
            )
            out_path = os.path.join(subdir, f"quad_{split_name}_idx_{idx:05d}.png")
            quad.save(out_path)
            done += 1
            if done % args.print_every == 0 or idx == i1 - 1:
                print(f"[{split_name}] {done}/{i1 - i0} (last idx {idx})")
        print(f"[done] {split_name}: {done} -> {subdir}")
        return done

    if args.all_splits:
        pairs = [
            ("train", str(_REPO_RPT / "data_train.npy"), str(_REPO_RPT / "mask_train.npy")),
            ("val", str(_REPO_RPT / "data_val.npy"), str(_REPO_RPT / "mask_val.npy")),
            ("test", str(_REPO_RPT / "data_test.npy"), str(_REPO_RPT / "mask_test.npy")),
        ]
        total = 0
        for split_name, dp, mp in pairs:
            total += loop_split(split_name, dp, mp, args.out_dir)
        print(f"[all_splits] total images: {total} -> {args.out_dir}")
        return

    if args.all_in_one:
        os.makedirs(args.out_dir, exist_ok=True)
        split_tag = Path(args.data_npy).stem.replace("data_", "")
        data = np.load(args.data_npy, mmap_mode="r")
        masks = np.load(args.mask_npy, mmap_mode="r")
        n = data.shape[0]
        assert masks.shape[0] == n
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
                args.threshold,
                args.gap,
                args.title_h,
                titles,
                oracle,
                args.col3_w_gt,
                args.col3_w_pr,
            )
            out_path = os.path.join(args.out_dir, f"quad_{split_tag}_idx_{idx:05d}.png")
            quad.save(out_path)
            done += 1
            if done % args.print_every == 0 or idx == i1 - 1:
                print(f"[{split_tag}] {done}/{i1 - i0}")
        print(f"[done] {done} -> {args.out_dir}")
        return

    data = np.load(args.data_npy, mmap_mode="r")
    masks = np.load(args.mask_npy, mmap_mode="r")
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
        args.threshold,
        args.gap,
        args.title_h,
        titles,
        oracle,
        args.col3_w_gt,
        args.col3_w_pr,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    quad.save(args.out)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
