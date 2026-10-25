#!/usr/bin/env python3
"""
Inference for one test_all_dataset model folder.

- mode=strip (default): vertical strips 512 / stride 512, each resized to 224, same per-strip
  normalization as RootBin predict_image (input_from_rgb_numpy, no global stats). Needs da_image + orig_image.

- mode=resize: whole image bilinear to input_h×input_w (default 224), same normalization; mask/blend at that size.

Run from repo root; each invocation should use a fresh Python process to avoid config import clashes.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def normalize_like_training(img_np, global_mean=None, global_std=None):
    img = img_np.astype(np.float32)
    mean = float(np.mean(img)) if global_mean is None else float(global_mean)
    std = float(np.std(img)) if global_std is None else float(global_std)
    if std < 1e-6:
        std = 1.0
    img = (img - mean) / std
    min_v = float(np.min(img))
    max_v = float(np.max(img))
    if max_v - min_v < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    img = (img - min_v) / (max_v - min_v) * 255.0
    return img.astype(np.float32)


def input_from_rgb_numpy(img_np, global_mean=None, global_std=None, no_preprocess=False):
    x = img_np.astype(np.float32)
    if no_preprocess:
        return np.clip(x, 0.0, 255.0)
    return normalize_like_training(x, global_mean=global_mean, global_std=global_std)


def _strip_x_coords(img_width, strip_width, stride):
    if img_width <= 0:
        return []
    if strip_width >= img_width:
        return [0]
    xs = list(range(0, img_width - strip_width + 1, stride))
    if len(xs) == 0 or xs[-1] + strip_width < img_width:
        xs.append(max(0, img_width - strip_width))
    return sorted(set(xs))


def _strip_overlap_weights(w_strip, h_strip, strip_blend):
    if strip_blend == "hann":
        wvec = np.hanning(max(1, int(w_strip))).astype(np.float32)
        return np.broadcast_to(wvec.reshape(1, -1), (int(h_strip), int(w_strip))).copy()
    if strip_blend == "mean":
        return np.ones((int(h_strip), int(w_strip)), dtype=np.float32)
    raise ValueError(f"strip_blend must be 'mean' or 'hann', got {strip_blend!r}")


def _strip_module_prefix(state_dict):
    """Strip DataParallel `module.` only (keep inner `model.` for UMambaBotAdapted)."""
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def _load_ckpt_state(weight_path):
    ckpt = torch.load(weight_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def _to_prob_numpy(pred_tensor):
    """BCHW -> HxW float prob in [0,1]."""
    y = pred_tensor
    if isinstance(y, (list, tuple)):
        y = y[0]
    if y.dim() == 4 and y.shape[1] > 1:
        y = y[:, 0:1]
    y = y.squeeze(1)
    arr = y.detach().float().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    mx, mn = float(arr.max()), float(arr.min())
    if mx > 1.5 or mn < -0.5:
        arr = torch.sigmoid(torch.from_numpy(arr)).numpy()
    return np.clip(arr, 0.0, 1.0)


def strip_predict(model, da_img: Image.Image, device, strip_width, strip_stride, train_hw, strip_blend="mean"):
    ori_w, ori_h = da_img.size
    pred_sum = np.zeros((ori_h, ori_w), dtype=np.float32)
    pred_wsum = np.zeros((ori_h, ori_w), dtype=np.float32)
    th, tw = train_hw
    xs = _strip_x_coords(ori_w, strip_width, strip_stride)
    model.eval()
    for x0 in xs:
        x1 = min(x0 + strip_width, ori_w)
        strip_img = da_img.crop((x0, 0, x1, ori_h))
        w_strip, h_strip = strip_img.size
        strip_rs = strip_img.resize((tw, th), Image.BILINEAR)
        img_np = input_from_rgb_numpy(np.array(strip_rs, dtype=np.float32), global_mean=None, global_std=None)
        x = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            pred = model(x)
            pred = F.interpolate(pred, size=(h_strip, w_strip), mode="bilinear", align_corners=False)
            p = _to_prob_numpy(pred)[:, :w_strip]
        weight = _strip_overlap_weights(w_strip, h_strip, strip_blend)
        pred_sum[:, x0:x1] += p * weight
        pred_wsum[:, x0:x1] += weight
    pred_wsum[pred_wsum == 0] = 1.0
    return pred_sum / pred_wsum


def resize224_predict(model, pil_img: Image.Image, device, train_hw):
    """Whole image bilinear to train_hw, same per-image normalization as strip path; prob map at train_hw."""
    th, tw = int(train_hw[0]), int(train_hw[1])
    img_rs = pil_img.resize((tw, th), Image.BILINEAR)
    img_np = input_from_rgb_numpy(np.array(img_rs, dtype=np.float32), global_mean=None, global_std=None)
    x = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
    model.eval()
    with torch.no_grad():
        pred = model(x)
        pred = F.interpolate(pred, size=(th, tw), mode="bilinear", align_corners=False)
        p = _to_prob_numpy(pred)
    return p, img_rs


def save_mask_blend(orig_img: Image.Image, pred, mask_path, blend_path, threshold, alpha=0.4):
    pred = np.clip(pred, 0.0, 1.0)
    mask = (pred >= threshold).astype(np.uint8) * 255
    Image.fromarray(mask, mode="L").save(mask_path)
    ori_np = np.array(orig_img, dtype=np.uint8)
    overlay = ori_np.copy()
    red = np.zeros_like(ori_np)
    red[..., 0] = 255
    fg = mask > 0
    overlay[fg] = (overlay[fg] * (1.0 - alpha) + red[fg] * alpha).astype(np.uint8)
    Image.fromarray(overlay).save(blend_path)


def build_model(model_root: Path, kind: str, device):
    sys.path.insert(0, str(model_root))
    from configs.config_setting import setting_config  # noqa: E402

    cfg = setting_config.model_config

    if kind == "unet++":
        from models.unetplusplus import UNetPlusPlus  # noqa: E402

        m = UNetPlusPlus(
            num_classes=cfg["num_classes"],
            input_channels=cfg["input_channels"],
            deep_supervision=cfg.get("deep_supervision", False),
        )
    elif kind == "unet":
        from models.unet import UNet  # noqa: E402

        m = UNet(num_classes=cfg["num_classes"], input_channels=cfg["input_channels"])
    elif kind == "deeplabv3":
        from models.deeplabv3plus import DeepLabV3Plus  # noqa: E402

        m = DeepLabV3Plus(
            num_classes=cfg["num_classes"],
            input_channels=cfg["input_channels"],
            output_stride=cfg["output_stride"],
        )
    elif kind == "segformer":
        from models.segformer import SegFormer  # noqa: E402

        m = SegFormer(
            num_classes=cfg["num_classes"],
            input_channels=cfg["input_channels"],
            embed_dims=cfg["embed_dims"],
            num_heads=cfg["num_heads"],
            mlp_ratios=cfg["mlp_ratios"],
            depths=cfg["depths"],
            sr_ratios=cfg["sr_ratios"],
            decoder_dim=cfg["decoder_dim"],
            drop_path_rate=cfg["drop_path_rate"],
        )
    elif kind == "umamba":
        from models.umamba_bot import UMambaBotAdapted  # noqa: E402

        m = UMambaBotAdapted(
            num_classes=cfg.get("num_classes", 1),
            input_channels=cfg.get("input_channels", 3),
        )
    elif kind == "iterroot":
        from models.iterroot import ITErRoot  # noqa: E402

        m = ITErRoot(
            num_classes=cfg["num_classes"],
            input_channels=cfg["input_channels"],
            base_channels=cfg["base_channels"],
            iters=cfg["iters"],
        )
    elif kind == "segroot":
        from models.segroot import SegNet  # noqa: E402

        m = SegNet(num_classes=cfg["num_classes"], input_channels=cfg["input_channels"])
    elif kind == "esegformer":
        from models.esegformer import ESegFormer  # noqa: E402

        m = ESegFormer(
            num_classes=cfg["num_classes"],
            input_channels=cfg["input_channels"],
            embedding_dim=cfg["embedding_dim"],
            embed_dims=cfg["embed_dims"],
            num_heads=cfg["num_heads"],
            mlp_ratios=cfg["mlp_ratios"],
            depths=cfg["depths"],
            sr_ratios=cfg["sr_ratios"],
            drop_path_rate=cfg["drop_path_rate"],
        )
    else:
        raise SystemExit(f"unknown kind {kind}")

    m = m.to(device)
    m.eval()
    return m


def detect_kind(name: str) -> str:
    """Folder names under test_all_dataset/<dataset>/…"""
    n = name.lower()
    if name == "unet++" or n == "unet++":
        return "unet++"
    if name == "unet":
        return "unet"
    if n == "deeplabv3":
        return "deeplabv3"
    if "esegformer" in n:
        return "esegformer"
    if n == "segformer":
        return "segformer"
    if "u-mamba" in n or "umamba" in n:
        return "umamba"
    if "iterroot" in n:
        return "iterroot"
    if "segroot" in n:
        return "segroot"
    raise SystemExit(f"cannot map folder name to model kind: {name!r}")


def main():
    ap = argparse.ArgumentParser(
        description="test_all_dataset model inference: strip (DA) or resize-224 whole image.",
    )
    ap.add_argument("--model_root", type=str, required=True)
    ap.add_argument("--weight", type=str, required=True)
    ap.add_argument(
        "--mode",
        type=str,
        default="strip",
        choices=["strip", "resize"],
        help="strip=512 vertical strips on da_image, blend on orig_image; resize=whole image to input_h×input_w then save at that size.",
    )
    ap.add_argument("--image", type=str, default="", help="resize mode: single RGB input path")
    ap.add_argument("--da_image", type=str, default="", help="strip mode: preprocessed / model-input style image")
    ap.add_argument("--orig_image", type=str, default="", help="strip mode: same size as da_image, used for blend output")
    ap.add_argument("--out_mask", type=str, required=True)
    ap.add_argument("--out_blend", type=str, required=True)
    ap.add_argument("--strip_width", type=int, default=512)
    ap.add_argument("--strip_stride", type=int, default=512)
    ap.add_argument("--input_h", type=int, default=224)
    ap.add_argument("--input_w", type=int, default=224)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--strip_blend", type=str, default="mean", choices=["mean", "hann"])
    args = ap.parse_args()

    model_root = Path(args.model_root).resolve()
    kind = detect_kind(model_root.name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_root, kind, device)
    sd = _strip_module_prefix(_load_ckpt_state(args.weight))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] missing_keys={len(missing)} sample={missing[:3]}")
    if unexpected:
        print(f"[warn] unexpected_keys={len(unexpected)} sample={unexpected[:3]}")

    os.makedirs(os.path.dirname(args.out_mask) or ".", exist_ok=True)
    train_hw = (args.input_h, args.input_w)

    if args.mode == "resize":
        if not args.image:
            raise SystemExit("resize mode requires --image")
        pil = Image.open(args.image).convert("RGB")
        pred, img_rs = resize224_predict(model, pil, device, train_hw)
        save_mask_blend(img_rs, pred, args.out_mask, args.out_blend, args.threshold)
        print(f"[ok] {model_root.name} resize224 -> {args.out_mask}")
        return

    if not args.da_image or not args.orig_image:
        raise SystemExit("strip mode requires --da_image and --orig_image")

    da_img = Image.open(args.da_image).convert("RGB")
    orig_img = Image.open(args.orig_image).convert("RGB")
    if da_img.size != orig_img.size:
        raise SystemExit(f"size mismatch da={da_img.size} orig={orig_img.size}")

    pred = strip_predict(
        model,
        da_img,
        device,
        args.strip_width,
        args.strip_stride,
        train_hw,
        strip_blend=args.strip_blend,
    )
    save_mask_blend(orig_img, pred, args.out_mask, args.out_blend, args.threshold)
    print(f"[ok] {model_root.name} -> {args.out_mask}")


if __name__ == "__main__":
    main()
