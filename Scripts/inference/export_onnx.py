#!/usr/bin/env python3
"""
Export RootExp (DA job01) to ONNX for C++ / ONNX Runtime inference.

Why export can still take several minutes:
  Mamba selective_scan is replaced by selective_scan_ref for ONNX; PyTorch 1.x then
  unrolls the sequential scan into a large graph (one-time). GPU speeds up forward/trace.

  python export_onnx.py              # default: cuda
  python export_onnx.py --device cpu
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHT = ROOT / "weights" / "da_job01_best-epoch322-loss0.1250.pth"
DEFAULT_OUT = ROOT / "weights" / "rootexp_da_224.onnx"

from predict_image import load_model  # noqa: E402


def _patch_selective_scan_for_export():
    import models.vmamba as vmamba_mod
    from models.selective_scan_onnx import selective_scan_export

    vmamba_mod.selective_scan_fn = selective_scan_export
    print("[export] selective_scan_fn -> selective_scan_ref (no CUDA custom op; ONNX-safe)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Export RootExp DA checkpoint to ONNX.")
    ap.add_argument("--weight", type=str, default=str(DEFAULT_WEIGHT))
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--drop_path_rate", type=float, default=0.05)
    ap.add_argument("--mlp_ratio", type=float, default=2.0)
    ap.add_argument("--hvss_d_state", type=int, default=8)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, falling back to CPU export", flush=True)
        args.device = "cpu"

    _patch_selective_scan_for_export()

    device = torch.device(args.device)
    t0 = time.time()
    print(f"[1/4] loading weights on {device} …", flush=True)
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
    model.eval()
    print(f"      loaded in {time.time()-t0:.1f}s", flush=True)

    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device)
    t1 = time.time()
    print("[2/4] warmup forward (builds scan graph once) …", flush=True)
    with torch.no_grad():
        _ = model(dummy)
    print(f"      warmup {time.time()-t1:.1f}s  output shape {tuple(_.shape)}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t2 = time.time()
    print("[3/4] torch.onnx.export — please wait (SSM loop → large ONNX graph) …", flush=True)
    sys.stdout.flush()
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["prob"],
            dynamic_axes=None,
            verbose=False,
        )
    print(f"      export done in {time.time()-t2:.1f}s", flush=True)

    t3 = time.time()
    print("[4/4] validating ONNX vs PyTorch …", flush=True)
    try:
        import onnx
        import onnxruntime as ort

        onnx.checker.check_model(onnx.load(str(out_path)))
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        x = dummy.detach().cpu().numpy()
        ort_out = sess.run(["prob"], {"input": x})[0]
        pt_out = model(dummy).detach().cpu().numpy()
        max_diff = float(np.max(np.abs(ort_out - pt_out)))
        print(f"      checker OK, max |ort-pt|={max_diff:.6e}", flush=True)
    except ImportError:
        print("      [warn] pip install onnx onnxruntime for validation", flush=True)
    except Exception as e:
        print(f"      [warn] validation skipped: {e}", flush=True)

    mb = out_path.stat().st_size / 1024 / 1024
    print(f"[ok] {out_path} ({mb:.1f} MB), total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
