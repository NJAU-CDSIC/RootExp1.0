#!/usr/bin/env python3
"""
For each model under test_all_dataset/RPT/<name>/ with a best*.pth or best.pth:
  whole-image resize to 224×224, predict, save mask + overlay at 224 (same path as strip_predict resize mode).

Default input: dataset/RPT/images/cotton_0618_7_7_4_8.png
Outputs: case studying/RPT/pred_testall_resize224_<image_stem>/
"""
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCR = REPO / "tools" / "strip_predict_testall_one_model.py"
RPT_BASE = REPO / "test_all_dataset" / "RPT"
DEFAULT_IMAGE = REPO / "dataset" / "RPT" / "images" / "cotton_0618_7_7_4_8.png"
PY = os.environ.get("PYTHON", "/home/young/anaconda3/envs/vmunet/bin/python")


def pick_best_pth(model_dir: Path):
    cands = sorted(glob.glob(str(model_dir / "best*.pth")))
    if not cands:
        return None

    def epoch_score(p):
        m = re.search(r"-epoch(\d+)-", os.path.basename(p), re.I)
        if m:
            return int(m.group(1))
        if os.path.basename(p).lower() == "best.pth":
            return 10**9
        return 0

    return max(cands, key=epoch_score)


def main():
    image = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_IMAGE
    if not image.is_file():
        raise SystemExit(f"missing image: {image}")

    stem = image.stem
    out_dir = REPO / "case studying" / "RPT" / f"pred_testall_resize224_{stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fail = []
    ok = 0
    for sub in sorted(RPT_BASE.iterdir()):
        if not sub.is_dir() or sub.name == "dataset":
            continue
        w = pick_best_pth(sub)
        if not w:
            print(f"[skip] no best*.pth in {sub.name}")
            fail.append((sub.name, "no weight"))
            continue
        safe = sub.name.replace("/", "_").replace(" ", "_")
        mask = out_dir / f"{safe}_mask.png"
        blend = out_dir / f"{safe}_overlay.png"
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
            "0.5",
        ]
        print(f"[run] {sub.name}  weight={os.path.basename(w)}")
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
