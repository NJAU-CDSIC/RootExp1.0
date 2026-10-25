# C++ / ONNX Runtime 推理说明

## 模型文件

| 文件 | 说明 |
|------|------|
| `weights/rootexp_da_224.onnx` | 固定输入 1×3×224×224，输出 1×1×224×224（sigmoid 概率） |
| `weights/da_job01_best-epoch322-loss0.1250.pth` | PyTorch 权重（Python 推理 / 重新导出 ONNX） |

## 单条 224×224 输入预处理（与 Python 一致）

对每条竖条 resize 到 224×224 后的 RGB 浮点图 `img`（HxWx3，0–255）：

1. `mean = mean(img)`，`std = std(img)`（整条 224×224 上统计）
2. `img = (img - mean) / std`
3. `img = (img - min) / (max - min) * 255`（若 max≈min 则置 0）
4. 送入 ONNX：`NCHW float32`

## 大图 strip 拼接（C++ 侧）

1. 原图 `img_1` 先做 CLAHE（见 `clahe_preprocess.py` / `docs/CLAHE.md`）
2. 沿 x 方向切竖条：宽 512、步长 512，高度 = 全图高
3. 每条 512×H → bilinear resize 到 224×224 → 预处理 → ONNX → 224×224 概率
4. 概率 bilinear 插值回 512×H，写入画布；重叠区取平均
5. 全图概率 ≥ **0.1** 二值化

详细流程与 Python 对齐见 `predict_image_DA_strip.md`。

## 重新导出 ONNX

```bash
python export_onnx.py              # 默认 GPU，约 15–20 分钟
python export_onnx.py --device cpu
```

导出日志示例：`docs/export_onnx_gpu.log`

## 依赖

- [ONNX Runtime](https://onnxruntime.ai/) C++ API
- OpenCV（CLAHE、resize、拼接）
