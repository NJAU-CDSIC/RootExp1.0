"""
Inference entrypoints for RootBin. Important for DA / machine_split job01:

  Training data are vertical strips (512 px wide, full height from crop_patches.py),
  each resized to 224×224 in Prepare_your_dataset.process_image — NOT 224×224 crops
  at native resolution.

  For correct big-image inference use --mode strip: per-strip resize to input_h×input_w,
  one forward pass, upsample logits back to strip size, blend strips (see predict_one_strip512).
  Use --mode strip_fusion to average probability maps from multiple strip geometries (e.g. 512:512 + 224:224).
  Use --strip_blend hann for Hann weights along strip width in overlaps (smoother than flat averaging).

  --mode tile slides windows on the full-resolution image; that geometry does NOT match
  the DA strip→224 resize pipeline. Details: predict_image_DA_strip.md
"""
import argparse
import os

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from models.rootbin import RootBinBlockMambaEAMASPPNet
from configs.config_setting import setting_config


def input_from_rgb_numpy(img_np, global_mean=None, global_std=None, no_preprocess=False):
    """Tensor-ready float32 HxWx3. If no_preprocess, expect pixels already match training (typically 0–255)."""
    x = img_np.astype(np.float32)
    if no_preprocess:
        return np.clip(x, 0.0, 255.0)
    return normalize_like_training(x, global_mean=global_mean, global_std=global_std)


def normalize_like_training(img_np, global_mean=None, global_std=None):
    img = img_np.astype(np.float32)
    if global_mean is None:
        mean = float(np.mean(img))
    else:
        mean = float(global_mean)
    if global_std is None:
        std = float(np.std(img))
    else:
        std = float(global_std)
    if std < 1e-6:
        std = 1.0
    img = (img - mean) / std
    min_v = float(np.min(img))
    max_v = float(np.max(img))
    if max_v - min_v < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    img = (img - min_v) / (max_v - min_v) * 255.0
    return img.astype(np.float32)


def load_train_stats(train_npy):
    arr = np.load(train_npy, mmap_mode="r")
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std < 1e-6:
        std = 1.0
    return mean, std


def load_model(weight_path, device, strict=False, overrides=None):
    cfg = dict(setting_config.model_config)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v
    model = RootBinBlockMambaEAMASPPNet(
        num_classes=cfg.get("num_classes", 1),
        input_channels=cfg.get("input_channels", 3),
        stage_name=cfg.get("stage_name", "stage0_unet_scblock_hvss_eam_aspp"),
        split_att=cfg.get("split_att", "fc"),
        encoder_channels=cfg.get("encoder_channels"),
        use_eam=cfg.get("use_eam", True),
        use_aspp=cfg.get("use_aspp", True),
        hvss_depth=cfg.get("hvss_depth", 1),
        drop_path_rate=cfg.get("drop_path_rate", 0.1),
        hvss_order=cfg.get("hvss_order", 2),
        hvss_d_state=cfg.get("hvss_d_state", 6),
        mlp_ratio=cfg.get("mlp_ratio", 1.5),
        use_skip_enhance=cfg.get("use_skip_enhance", True),
        skip_fusion_dim=cfg.get("skip_fusion_dim", 64),
    ).to(device)

    ckpt = torch.load(weight_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    load_result = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        missing = getattr(load_result, "missing_keys", [])
        unexpected = getattr(load_result, "unexpected_keys", [])
        print(f"[load_state_dict] missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
        if missing:
            print(f"[load_state_dict] missing sample: {missing[:5]}")
        if unexpected:
            print(f"[load_state_dict] unexpected sample: {unexpected[:5]}")
    model.eval()
    return model


def _save_mask_overlay(ori_img, pred, out_mask_path, out_overlay_path, threshold):
    pred = np.clip(pred, 0.0, 1.0)
    mask = (pred >= threshold).astype(np.uint8) * 255
    Image.fromarray(mask, mode="L").save(out_mask_path)
    ori_np = np.array(ori_img, dtype=np.uint8)
    overlay = ori_np.copy()
    red = np.zeros_like(ori_np)
    red[..., 0] = 255
    fg = mask > 0
    overlay[fg] = (overlay[fg] * 0.6 + red[fg] * 0.4).astype(np.uint8)
    Image.fromarray(overlay).save(out_overlay_path)


def _tile_starts(total, tile, stride):
    if total <= tile:
        return [0]
    starts = list(range(0, total - tile + 1, stride))
    if starts[-1] != total - tile:
        starts.append(total - tile)
    return starts


def _strip_x_coords(img_width, strip_width, stride):
    """Same geometry as dataset/da/crop_patches.py: vertical strips, x offsets along width."""
    if img_width <= 0:
        return []
    if strip_width >= img_width:
        return [0]
    xs = list(range(0, img_width - strip_width + 1, stride))
    if len(xs) == 0 or xs[-1] + strip_width < img_width:
        xs.append(max(0, img_width - strip_width))
    return sorted(set(xs))


def _strip_overlap_weights(w_strip, h_strip, strip_blend):
    """Per-pixel weights on a vertical strip (H×W). Hann emphasizes strip center, softer seams."""
    if strip_blend == "hann":
        wvec = np.hanning(max(1, int(w_strip))).astype(np.float32)
        weight = np.broadcast_to(wvec.reshape(1, -1), (int(h_strip), int(w_strip))).copy()
    elif strip_blend == "mean":
        weight = np.ones((int(h_strip), int(w_strip)), dtype=np.float32)
    else:
        raise ValueError(f"strip_blend must be 'mean' or 'hann', got {strip_blend!r}")
    return weight


def predict_one_strip512(
    model,
    ori_img,
    device,
    strip_width,
    strip_stride,
    train_in_hw=(224, 224),
    global_mean=None,
    global_std=None,
    no_preprocess=False,
    strip_blend="mean",
):
    """
    DA / job01 strip inference aligned with da_cropped Prepare_your_dataset.process_image:
    crop vertical strips (default 512 px wide, full native height), resize each strip to
    train_in_hw (224×224), run the network once per strip, upsample logits back to strip
    size and blend into the full canvas (same geometry as crop_patches strip stepping).
    Overlap blending: strip_blend='mean' (uniform) or 'hann' (Hann along strip width).
    """
    ori_w, ori_h = ori_img.size
    pred_sum = np.zeros((ori_h, ori_w), dtype=np.float32)
    pred_wsum = np.zeros((ori_h, ori_w), dtype=np.float32)
    xs = _strip_x_coords(ori_w, strip_width, strip_stride)
    th, tw = train_in_hw
    for x0 in xs:
        x1 = min(x0 + strip_width, ori_w)
        strip_img = ori_img.crop((x0, 0, x1, ori_h))
        w_strip, h_strip = strip_img.size
        strip_rs = strip_img.resize((tw, th), Image.BILINEAR)
        img_np = input_from_rgb_numpy(
            np.array(strip_rs, dtype=np.float32),
            global_mean=global_mean,
            global_std=global_std,
            no_preprocess=no_preprocess,
        )
        x = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            pred = model(x)
            pred = pred if not isinstance(pred, tuple) else pred[0]
            pred = F.interpolate(pred, size=(h_strip, w_strip), mode="bilinear", align_corners=False)
            p = pred.squeeze().detach().cpu().numpy()
        weight = _strip_overlap_weights(w_strip, h_strip, strip_blend)
        pred_sum[:, x0:x1] += p[:, :w_strip] * weight
        pred_wsum[:, x0:x1] += weight
    pred_wsum[pred_wsum == 0] = 1.0
    return pred_sum / pred_wsum


def predict_one_strip_fusion(
    model,
    ori_img,
    device,
    strip_specs,
    train_in_hw=(224, 224),
    global_mean=None,
    global_std=None,
    no_preprocess=False,
    strip_blend="mean",
):
    """Run strip inference for each (strip_width, strip_stride), same train_in_hw resize per strip; fuse by mean prob."""
    preds = []
    for strip_width, strip_stride in strip_specs:
        p = predict_one_strip512(
            model,
            ori_img,
            device,
            strip_width=strip_width,
            strip_stride=strip_stride,
            train_in_hw=train_in_hw,
            global_mean=global_mean,
            global_std=global_std,
            no_preprocess=no_preprocess,
            strip_blend=strip_blend,
        )
        preds.append(p)
    return np.mean(np.stack(preds, axis=0), axis=0)


def predict_one_resize(
    model,
    ori_img,
    device,
    size_hw=(224, 224),
    global_mean=None,
    global_std=None,
    no_preprocess=False,
    upscale_to_original=True,
):
    """Resize to size_hw, forward, then either upsample logits to original image size or keep input size."""
    ori_w, ori_h = ori_img.size
    img_rs = ori_img.resize((size_hw[1], size_hw[0]), Image.BILINEAR)
    img_np = input_from_rgb_numpy(
        np.array(img_rs, dtype=np.float32),
        global_mean=global_mean,
        global_std=global_std,
        no_preprocess=no_preprocess,
    )
    x = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred = model(x)
        pred = pred if not isinstance(pred, tuple) else pred[0]
        if upscale_to_original:
            pred = F.interpolate(pred, size=(ori_h, ori_w), mode="bilinear", align_corners=False)
        else:
            ph, pw = int(size_hw[0]), int(size_hw[1])
            if pred.shape[-2] != ph or pred.shape[-1] != pw:
                pred = F.interpolate(pred, size=(ph, pw), mode="bilinear", align_corners=False)
        pred_np = pred.squeeze().detach().cpu().numpy()
    overlay_base = ori_img if upscale_to_original else img_rs
    return pred_np, overlay_base


def predict_one_tiled(
    model,
    ori_img,
    device,
    tile_hw=(128, 128),
    stride_hw=(96, 96),
    global_mean=None,
    global_std=None,
    no_preprocess=False,
):
    ori_np = np.array(ori_img, dtype=np.float32)
    h, w = ori_np.shape[:2]
    tile_h, tile_w = tile_hw
    stride_h, stride_w = stride_hw

    pred_sum = np.zeros((h, w), dtype=np.float32)
    pred_cnt = np.zeros((h, w), dtype=np.float32)
    ys = _tile_starts(h, tile_h, stride_h)
    xs = _tile_starts(w, tile_w, stride_w)

    for y in ys:
        for x0 in xs:
            crop = ori_np[y:y + tile_h, x0:x0 + tile_w]
            ch, cw = crop.shape[:2]
            if ch < tile_h or cw < tile_w:
                pad_h = tile_h - ch
                pad_w = tile_w - cw
                crop = np.pad(crop, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

            crop = input_from_rgb_numpy(
                crop, global_mean=global_mean, global_std=global_std, no_preprocess=no_preprocess
            )
            tx = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float().to(device)
            with torch.no_grad():
                p = model(tx)
                p = p if not isinstance(p, tuple) else p[0]
                p = p.squeeze().detach().cpu().numpy()

            pred_sum[y:y + ch, x0:x0 + cw] += p[:ch, :cw]
            pred_cnt[y:y + ch, x0:x0 + cw] += 1.0

    pred_cnt[pred_cnt == 0] = 1.0
    return pred_sum / pred_cnt


def predict_one_from_npy_slice(
    model,
    img_np,
    out_mask_path,
    out_overlay_path,
    device,
    size_hw=(224, 224),
    threshold=0.5,
    global_mean=None,
    global_std=None,
    no_preprocess=False,
):
    """Predict one sample from raw npy slice (H,W,3), same normalization as file-based path."""
    ori_np_display = np.clip(img_np.astype(np.float64), 0.0, 255.0).astype(np.uint8)
    ori_img = Image.fromarray(ori_np_display, mode="RGB")
    ori_h, ori_w = ori_np_display.shape[:2]
    img_work = img_np.astype(np.float32)
    if (ori_h, ori_w) != (size_hw[0], size_hw[1]):
        pil_rs = ori_img.resize((size_hw[1], size_hw[0]), Image.BILINEAR)
        img_work = np.array(pil_rs, dtype=np.float32)
    img_norm = input_from_rgb_numpy(
        img_work, global_mean=global_mean, global_std=global_std, no_preprocess=no_preprocess
    )
    x = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred = model(x)
        pred = pred if not isinstance(pred, tuple) else pred[0]
        pred = F.interpolate(pred, size=(ori_h, ori_w), mode="bilinear", align_corners=False)
        pred = pred.squeeze().detach().cpu().numpy()
    _save_mask_overlay(ori_img, pred, out_mask_path, out_overlay_path, threshold)


def predict_one(
    model,
    image_path,
    out_mask_path,
    out_overlay_path,
    device,
    size_hw=(224, 224),
    threshold=0.5,
    mode="resize",
    tile_hw=(128, 128),
    stride_hw=(96, 96),
    strip_width=512,
    strip_stride=512,
    strip_specs=None,
    strip_blend="mean",
    global_mean=None,
    global_std=None,
    no_preprocess=False,
    at_input_size=False,
):
    ori_img = Image.open(image_path).convert("RGB")
    overlay_base = ori_img
    if mode == "tile":
        pred = predict_one_tiled(
            model,
            ori_img,
            device,
            tile_hw=tile_hw,
            stride_hw=stride_hw,
            global_mean=global_mean,
            global_std=global_std,
            no_preprocess=no_preprocess,
        )
    elif mode == "strip":
        pred = predict_one_strip512(
            model,
            ori_img,
            device,
            strip_width=strip_width,
            strip_stride=strip_stride,
            train_in_hw=(size_hw[0], size_hw[1]),
            global_mean=global_mean,
            global_std=global_std,
            no_preprocess=no_preprocess,
            strip_blend=strip_blend,
        )
    elif mode == "strip_fusion":
        specs = strip_specs if strip_specs else [(512, 512), (224, 224)]
        pred = predict_one_strip_fusion(
            model,
            ori_img,
            device,
            strip_specs=specs,
            train_in_hw=(size_hw[0], size_hw[1]),
            global_mean=global_mean,
            global_std=global_std,
            no_preprocess=no_preprocess,
            strip_blend=strip_blend,
        )
    else:
        pred, overlay_base = predict_one_resize(
            model,
            ori_img,
            device,
            size_hw=size_hw,
            global_mean=global_mean,
            global_std=global_std,
            no_preprocess=no_preprocess,
            upscale_to_original=not at_input_size,
        )
    _save_mask_overlay(overlay_base, pred, out_mask_path, out_overlay_path, threshold)


def parse_args():
    p = argparse.ArgumentParser(description="RootBin_Block_Mamba_EAM_ASPP prediction")
    p.add_argument("--image", type=str, default="", help="Single image path")
    p.add_argument("--input_dir", type=str, default="", help="Run on all .jpg/.jpeg/.png under this folder")
    p.add_argument(
        "--data_npy",
        type=str,
        default="",
        help="NxHxWx3 npy (e.g. data_train.npy): predict slices without reading PNG folder",
    )
    p.add_argument("--npy_max", type=int, default=5, help="Max samples when --data_npy is set")
    p.add_argument("--npy_start", type=int, default=0, help="First index in npy")
    p.add_argument("--weight", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./pred_outputs")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--input_h", type=int, default=224)
    p.add_argument("--input_w", type=int, default=224)
    p.add_argument(
        "--mode",
        type=str,
        default="resize",
        choices=["resize", "tile", "strip", "strip_fusion"],
        help=(
            "strip = vertical strips (crop_patches geometry); each strip resize to --input_h x --input_w "
            "(224×224 like Prepare_your_dataset), infer once, upsample pred back — not native-resolution tiling. "
            "strip_fusion = multiple strip geometries, mean-fuse probability maps (see --strip_fusion)."
        ),
    )
    p.add_argument(
        "--strip_fusion",
        type=str,
        default="512:512,224:224",
        help="For --mode strip_fusion: comma-separated width:stride pairs, e.g. 512:512,224:224",
    )
    p.add_argument("--strip_width", type=int, default=512, help="Vertical strip width for --mode strip")
    p.add_argument("--strip_stride", type=int, default=512, help="Strip stride along x for --mode strip")
    p.add_argument(
        "--strip_blend",
        type=str,
        default="mean",
        choices=["mean", "hann"],
        help="Strip overlap blend: mean=flat average; hann=Hann weight along strip width (smoother seams).",
    )
    p.add_argument(
        "--tile_h",
        type=int,
        default=128,
        help="Tile height for --mode tile (default 128; model was often trained at 224—override if needed).",
    )
    p.add_argument("--tile_w", type=int, default=128)
    p.add_argument(
        "--stride_h",
        type=int,
        default=96,
        help="Stride height for --mode tile (default 96 ≈25%% overlap with 128 tile, same ratio as 168/224).",
    )
    p.add_argument("--stride_w", type=int, default=96)
    p.add_argument("--train_npy", type=str, default="", help="Optional data_train.npy path for global mean/std")
    p.add_argument(
        "--no_preprocess",
        action="store_true",
        help="Skip normalize_like_training; feed float32 RGB in [0,255] as-is (image already preprocessed like training).",
    )
    p.add_argument(
        "--at_input_size",
        action="store_true",
        help="For --mode resize only: save mask/overlay at --input_h x --input_w (e.g. 224×224) instead of original image size.",
    )
    p.add_argument("--strict", action="store_true", help="Use strict=True when loading weights")
    p.add_argument("--drop_path_rate", type=float, default=None, help="Override model_config (match training job)")
    p.add_argument("--mlp_ratio", type=float, default=None, help="Override model_config (match training job)")
    p.add_argument("--hvss_d_state", type=int, default=None, help="Override model_config (match training job)")
    return p.parse_args()


def _parse_strip_fusion(spec_str):
    """Parse '512:512,224:224' -> [(512,512), (224,224)]."""
    out = []
    for part in spec_str.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"strip_fusion segment must be width:stride, got {part!r}")
        w, s = part.split(":", 1)
        out.append((int(w.strip()), int(s.strip())))
    if not out:
        raise ValueError("strip_fusion is empty")
    return out


def _collect_images(args):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if args.input_dir:
        paths = []
        for fn in sorted(os.listdir(args.input_dir)):
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts:
                paths.append(os.path.join(args.input_dir, fn))
        return paths
    if args.image:
        return [args.image]
    return []


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats_path = "" if args.no_preprocess else (args.train_npy or (args.data_npy if args.data_npy else ""))
    global_mean = None
    global_std = None
    if stats_path:
        global_mean, global_std = load_train_stats(stats_path)
        print(f"[train_stats] from={stats_path} mean={global_mean:.6f}, std={global_std:.6f}")
    elif args.no_preprocess:
        print("[input] no_preprocess: using RGB float32 [0,255] without normalize_like_training")
    overrides = {
        "drop_path_rate": args.drop_path_rate,
        "mlp_ratio": args.mlp_ratio,
        "hvss_d_state": args.hvss_d_state,
    }
    model = load_model(args.weight, device, strict=args.strict, overrides=overrides)

    if args.data_npy:
        arr = np.load(args.data_npy, mmap_mode="r")
        n_total = arr.shape[0]
        i0 = max(0, args.npy_start)
        i1 = min(i0 + args.npy_max, n_total)
        if i0 >= n_total:
            raise SystemExit(f"npy_start {i0} >= N {n_total}")
        print(f"[data_npy] {args.data_npy} shape={arr.shape} indices [{i0}, {i1})")
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
            print(f"[ok] npy index {idx}")
        print(f"Done. {i1 - i0} npy slice(s).")
        return

    paths = _collect_images(args)
    if not paths:
        raise SystemExit("Provide --data_npy, or --image, or a non-empty --input_dir with images.")
    if args.at_input_size and args.mode != "resize":
        print("[warn] --at_input_size only applies to --mode resize; ignoring")
        args.at_input_size = False
    fusion_specs = _parse_strip_fusion(args.strip_fusion) if args.mode == "strip_fusion" else None
    if args.mode == "strip_fusion":
        print(
            f"[strip_fusion] specs={fusion_specs} train_in={args.input_h}x{args.input_w} "
            f"fuse=mean strip_blend={args.strip_blend}"
        )
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
            at_input_size=args.at_input_size,
        )
        print(f"[ok] {image_path}")
    print(f"Done. {len(paths)} image(s).")


if __name__ == "__main__":
    main()
