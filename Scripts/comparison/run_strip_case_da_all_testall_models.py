#!/usr/bin/env python3
"""
For every test_all_dataset/<dataset>/<model>/ with a best*.pth:
  strip 512 / stride 512 / 224 on DA preprocessed 5-1_fan_.jpg, blend on img_1 original.
Writes to case studying/da/ as {dataset}__{model}_mask.png and _blending.png
"""
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCR = REPO / "tools" / "strip_predict_testall_one_model.py"
OUT = REPO / "case studying" / "da"
DA = REPO / "dataset" / "da" / "images" / "5-1_fan_.jpg"
ORIG = Path("/home/young/桌面/rape_root_seg/dataset/img_1/5-1_fan_.JPG")
PY = os.environ.get(
    "PYTHON",
    "/home/young/anaconda3/envs/vmunet/bin/python",
)


def pick_best_pth(model_dir: str):
    cands = sorted(glob.glob(os.path.join(model_dir, "best*.pth")))
    if not cands:
        return None

    def epoch_score(p):
        m = re.search(r"-epoch(\d+)-", os.path.basename(p), re.I)
        if m:
            return int(m.group(1))
        if os.path.basename(p).lower() == "best.pth":
            return -1
        return 0

    return max(cands, key=epoch_score)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fail = []
    ok = 0
    for ds in ("da", "prim", "RPT", "sumilte", "wheat"):
        base = REPO / "test_all_dataset" / ds
        if not base.is_dir():
            print(f"[skip] no {base}")
            continue
        for sub in sorted(base.iterdir()):
            if not sub.is_dir() or sub.name == "dataset":
                continue
            w = pick_best_pth(str(sub))
            if not w:
                print(f"[skip] no best*.pth {sub}")
                fail.append((str(sub), "no weight"))
                continue
            stem = f"{ds}__{sub.name.replace('/', '_')}"
            mask = OUT / f"{stem}_mask.png"
            blend = OUT / f"{stem}_blending.png"
            cmd = [
                PY,
                str(SCR),
                "--model_root",
                str(sub),
                "--weight",
                w,
                "--da_image",
                str(DA),
                "--orig_image",
                str(ORIG),
                "--out_mask",
                str(mask),
                "--out_blend",
                str(blend),
                "--strip_width",
                "512",
                "--strip_stride",
                "512",
                "--input_h",
                "224",
                "--input_w",
                "224",
                "--threshold",
                "0.1",
                "--strip_blend",
                "mean",
            ]
            print(f"[run] {stem}  weight={os.path.basename(w)}")
            r = subprocess.run(cmd, cwd=str(REPO))
            if r.returncode != 0:
                fail.append((stem, f"exit {r.returncode}"))
            else:
                ok += 1
    print(f"\nDone ok={ok} fail={len(fail)}")
    for item, reason in fail:
        print(f"  FAIL {item}: {reason}")


if __name__ == "__main__":
    main()
