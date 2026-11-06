#!/usr/bin/env python3
"""
End-to-end BnaRoot panorama demo: field photo -> CLAHE -> strip512 RootExp -> mask + overlay.

  python run_panorama_demo.py \
    --field_image /path/to/field.jpg \
    --out_dir ./out
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PKG = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from clahe_preprocess import process_one  # noqa: E402
from predict_bnaroot_strip import main as predict_main  # noqa: E402

DEFAULT_WEIGHT = PKG / "RootExp" / "Pre-trained" / "best.pt"


def main():
    ap = argparse.ArgumentParser(description="CLAHE + strip RootExp on one field panorama")
    ap.add_argument("--field_image", type=str, required=True, help="Original field RGB/JPG")
    ap.add_argument("--out_dir", type=str, default=str(ROOT / "pred_outputs"))
    ap.add_argument("--clahe_dir", type=str, default="", help="Reuse/save CLAHE RGB here (default: temp)")
    ap.add_argument("--weight", type=str, default=str(DEFAULT_WEIGHT))
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--strip_width", type=int, default=512)
    ap.add_argument("--strip_stride", type=int, default=512)
    args, _unknown = ap.parse_known_args()

    field = Path(args.field_image).resolve()
    if not field.is_file():
        sys.exit(f"Missing field image: {field}")

    if args.clahe_dir.strip():
        clahe_dir = Path(args.clahe_dir)
        clahe_dir.mkdir(parents=True, exist_ok=True)
        clahe_path = clahe_dir / (field.stem + ".jpg")
        cleanup = False
    else:
        clahe_path = Path(tempfile.mkdtemp(prefix="clahe_")) / (field.stem + ".jpg")
        cleanup = True

    print(f"[1/2] CLAHE {field.name} …")
    if not process_one(field, clahe_path, clip_limit=2.0, grid_size=(8, 8)):
        sys.exit("CLAHE failed")

    print("[2/2] strip predict …")
    os.makedirs(args.out_dir, exist_ok=True)
    argv = [
        "predict_bnaroot_strip.py",
        "--input_image",
        str(clahe_path),
        "--viz_image",
        str(field),
        "--weight",
        args.weight,
        "--out_dir",
        args.out_dir,
        "--threshold",
        str(args.threshold),
        "--strip_width",
        str(args.strip_width),
        "--strip_stride",
        str(args.strip_stride),
    ]
    sys.argv = argv
    predict_main()

    if cleanup:
        try:
            clahe_path.unlink(missing_ok=True)
            clahe_path.parent.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
