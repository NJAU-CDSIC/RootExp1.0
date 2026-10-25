#!/usr/bin/env python3
"""
For each model under test_all_dataset/sumilte/<name>/ with best*.pth:
  resize-224 inference on one RGB image (default: case studying/sumlite train-00000 original).

Writes mask + blending per model. Copies input once as reference_original.png.

Threshold: env SUMILTE_TESTALL_THRESHOLD (default 0.5).
"""
import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCR = REPO / "tools" / "strip_predict_testall_one_model.py"
SUM_BASE = REPO / "test_all_dataset" / "sumilte"
DEFAULT_IMAGE = (
    REPO / "case studying" / "sumlite" / "sumilte_train_idx_00000_original_rgb.png"
)
PY = os.environ.get("PYTHON", "/home/young/anaconda3/envs/vmunet/bin/python")


def pick_best_pth(model_dir: Path):
    """Prefer highest `best-epochNNN-*.pth` when present; else first best*.pth."""
    cands = sorted(glob.glob(str(model_dir / "best*.pth")))
    if not cands:
        return None
    epoch_cands = [c for c in cands if re.search(r"-epoch\d+-", os.path.basename(c), re.I)]

    def epoch_num(p):
        m = re.search(r"-epoch(\d+)-", os.path.basename(p), re.I)
        return int(m.group(1)) if m else -1

    if epoch_cands:
        return max(epoch_cands, key=epoch_num)
    return cands[0]


def main():
    image = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_IMAGE
    if not image.is_file():
        raise SystemExit(f"missing image: {image}")

    th = float(os.environ.get("SUMILTE_TESTALL_THRESHOLD", "0.5"))
    stem = image.stem
    out_dir = REPO / "case studying" / "sumlite" / f"pred_testall_resize224_{stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_copy = out_dir / "reference_original.png"
    if not ref_copy.exists() or ref_copy.resolve() != image.resolve():
        shutil.copy2(image, ref_copy)

    fail = []
    ok = 0
    for sub in sorted(SUM_BASE.iterdir()):
        if not sub.is_dir() or sub.name == "dataset":
            continue
        w = pick_best_pth(sub)
        if not w:
            print(f"[skip] no best*.pth in {sub.name}")
            fail.append((sub.name, "no weight"))
            continue
        safe = sub.name.replace("/", "_").replace(" ", "_")
        mask = out_dir / f"{safe}_mask.png"
        blend = out_dir / f"{safe}_blending.png"
        cmd = [
            PY,
            str(SCR),
            "--mode",
            "resize",
            "--model_root",
            str(sub.resolve()),
            "--weight",
            w,
            "--image",
            str(image),
            "--out_mask",
            str(mask),
            "--out_blend",
            str(blend),
            "--input_h",
            "224",
            "--input_w",
            "224",
            "--threshold",
            str(th),
        ]
        print(f"[run] {sub.name}  weight={os.path.basename(w)}  th={th}")
        r = subprocess.run(cmd, cwd=str(REPO))
        if r.returncode != 0:
            fail.append((sub.name, f"exit {r.returncode}"))
        else:
            ok += 1

    print(f"\nDone ok={ok} fail={len(fail)}  out_dir={out_dir}")
    for name, reason in fail:
        print(f"  FAIL {name}: {reason}")


if __name__ == "__main__":
    main()
