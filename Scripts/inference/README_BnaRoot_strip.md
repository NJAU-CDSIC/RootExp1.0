# BnaRoot strip inference

CLAHE preprocess + vertical strip inference (512 → 224×224 → stitch) for large BnaRoot field images, plus optional ONNX export.

## Main scripts

| Script | Role |
|--------|------|
| `clahe_preprocess.py` | Grayscale + CLAHE → 3-channel RGB |
| `predict_bnaroot_strip.py` | Strip512→224 RootExp inference |
| `run_panorama_demo.py` | Field photo → CLAHE → strip → mask/overlay |
| `export_onnx.py` | Export RootExp checkpoint to ONNX |
| `predict_image.py` | Low-level resize / strip / tile entrypoints |

Default weight: `../../RootExp/Pre-trained/best.pt`

## Example

```bash
python run_panorama_demo.py \
  --field_image /path/to/field.jpg \
  --out_dir ./out
```

See also `docs/CLAHE.md` and `docs/ONNX_CPP.md`.
