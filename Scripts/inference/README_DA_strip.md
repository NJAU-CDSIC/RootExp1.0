# DA 数据集 RootExp 便携推理包

用于在另一台机器上对 DA 油菜大图做 **CLAHE 预处理 + strip 竖条滑动推理**（512 px → 224×224 → 拼接），或导出/使用 **ONNX** 供 C++ 调用。

---

## 目录结构（完整）

```
DA_strip_inference_portable/
├── README.md                          # 本说明
├── requirements.txt                   # Python 依赖
├── .gitignore
│
├── weights/
│   ├── da_job01_best-epoch322-loss0.1250.pth   # DA job01 PyTorch 权重 (~90MB)
│   └── rootexp_da_224.onnx                     # ONNX 224×224 单条输入 (~95MB)
│
├── clahe_preprocess.py                # CLAHE（同 rape_root_seg/dataset/2.py）
├── run_panorama_demo.py               # CLAHE + strip 一键大图
├── predict_da_strip_one.py            # 单张 CLAHE 图 strip 推理
├── predict_image.py                   # 通用推理（单图/文件夹）
├── export_onnx.py                     # 导出 ONNX（默认 GPU）
├── demo_img1_strip_predict_one.py     # 原仓库 demo（含 UNet++，可选）
│
├── configs/config_setting.py          # 模型配置（推理用）
├── models/                            # RootExp 网络结构
│   ├── rootbin.py
│   ├── eam_module.py, sc_att_bridge.py, hvss_modules.py, vmamba.py, ...
│   └── selective_scan_onnx.py         # ONNX 导出时替换 Mamba CUDA op
│
├── docs/
│   ├── CLAHE.md                       # CLAHE 预处理说明
│   ├── ONNX_CPP.md                    # C++ / ONNX Runtime 集成
│   └── export_onnx_gpu.log            # 本次 GPU 导出日志
│
├── predict_image_DA_strip.md          # strip 推理原理（勿与 tile 混淆）
│
└── examples/5-1_fan_/                 # 本地验证样例（5-1_fan_.JPG）
    ├── clahe.jpg                      # CLAHE 预处理结果
    ├── mask.png                       # 预测掩膜
    └── overlay.png                    # 原图叠加可视化
```

---

## 环境

```bash
pip install -r requirements.txt
# PyTorch 推理若报 Mamba 相关错误：
pip install mamba-ssm causal-conv1d
```

建议：Python 3.8+，PyTorch 1.13+cu117（与训练一致）。

---

## 快速使用

### 1. 大图完整流程（推荐）

```bash
python run_panorama_demo.py \
  --field_image /path/to/img_1/5-1_fan_.JPG \
  --out_dir ./out
```

### 2. 分步：CLAHE → 推理

```bash
python clahe_preprocess.py \
  --image /path/to/img_1/5-1_fan_.JPG \
  --out ./clahe/5-1_fan_.jpg

python predict_da_strip_one.py \
  --da_image ./clahe/5-1_fan_.jpg \
  --viz_image /path/to/img_1/5-1_fan_.JPG \
  --out_dir ./out \
  --threshold 0.1
```

### 3. 批量 strip 推理

```bash
python predict_image.py \
  --input_dir /path/to/clahe_images \
  --weight weights/da_job01_best-epoch322-loss0.1250.pth \
  --mode strip \
  --strip_width 512 --strip_stride 512 \
  --input_h 224 --input_w 224 \
  --threshold 0.1 \
  --drop_path_rate 0.05 --mlp_ratio 2.0 --hvss_d_state 8 \
  --out_dir ./out
```

---

## 超参与阈值

| 参数 | 值 |
|------|-----|
| `drop_path_rate` | 0.05 |
| `mlp_ratio` | 2.0 |
| `hvss_d_state` | 8 |
| `threshold` | 0.1 |
| strip | 512 / stride 512 |

---

## 重要说明

1. **网络输入**：CLAHE 预处理图（`clahe_preprocess.py`），**不要**直接把 `img_1` 原图喂网络。
2. **可视化**：掩膜叠加在 `img_1` 原图上（`--viz_image`）。
3. **勿用 `--mode tile`**：与训练 strip→224 几何不一致，见 `predict_image_DA_strip.md`。
4. **本地验证**：`examples/5-1_fan_/` 与历史结果 IoU ≈ 0.998。

---

## C++ / ONNX

- 模型：`weights/rootexp_da_224.onnx`
- 集成说明：`docs/ONNX_CPP.md`
- 重新导出：`python export_onnx.py`（默认 GPU，约 15–20 min）

---

## 参考路径（本机）

| 用途 | 路径 |
|------|------|
| CLAHE 原脚本 | `/home/young/桌面/rape_root_seg/dataset/2.py` |
| 田间原图 | `rape_root_seg/dataset/img_1/` |
| 训练用 CLAHE 图 | `testGeModel/dataset/da/images/` |

---

打包日期：2026-07-04
