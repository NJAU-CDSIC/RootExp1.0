# CLAHE 预处理说明

与 `rape_root_seg/dataset/2.py` 逻辑一致：

1. BGR 读图（支持中文路径）
2. 转灰度
3. CLAHE：`clipLimit=2.0`，`tileGridSize=(8, 8)`
4. 灰度复制为 RGB 三通道保存（与 `dataset/da/images` 训练分布一致）

## 命令

```bash
# 单张
python clahe_preprocess.py \
  --image /path/to/img_1/5-1_fan_.JPG \
  --out ./clahe/5-1_fan_.jpg

# 批量
python clahe_preprocess.py \
  --input_dir /path/to/img_1 \
  --out_dir ./clahe_out
```

**注意**：网络输入用 CLAHE 图；可视化叠加用未预处理的 `img_1` 原图。
