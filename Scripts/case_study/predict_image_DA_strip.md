# DA / job01 大图推理说明（勿与通用 tile 混淆）

## 训练数据实际长什么样

- 大图先按 `dataset/da/crop_patches.py` 切成 **竖条**：宽 **512**、高 **整图高度**，沿 **x** 方向步进（默认 stride **512**），右边缘会对齐最后一条。
- `dataset/da/da_cropped/Prepare_your_dataset.py` 里 **`process_image`**：读入每一条图后 **`resize((224, 224), BILINEAR)`**，再进 npy / `dataset_normalized` 训练。

也就是说：**网络看到的是「整条竖带压成 224×224」**，不是在原图分辨率上裁 **224×224** 滑窗。

## 推理应该怎么配（正确）

使用 **`predict_image.py --mode strip`**：

1. 与原脚本相同的竖条几何：`--strip_width 512`、`--strip_stride 512`（可按数据准备改动）。
2. 每一条 **`512 × H`**（或右缘较窄条）**整张 resize 到 `--input_h × --input_w`（默认 224×224）**，**一次前向**。
3. 将输出的 logits **双线性插值回该条在原图上的 `H × 条宽`**，写回画布；条与条重叠区域 **按覆盖次数平均**。

示例：

```bash
python predict_image.py --image "/path/to/panorama.jpg" --weight "/path/to/job01_best.pth" \
  --mode strip --strip_width 512 --strip_stride 512 \
  --input_h 224 --input_w 224 \
  --threshold 0.1 --out_dir "./out"
```

可选：加 `--train_npy /path/to/data_train.npy` 以贴近 loader 里全局 mean/std（仍配合每条 resize 后的归一化逻辑）。

## 常见错误（错误）

- **`--mode tile`**：在**原图分辨率**上铺 **224×224（或其它）**小块滑动。这与「整条 512×H 先压成 224×224」的训练尺度不一致，**不适合当作 DA job01 的标准大图推理**。
- **`--mode strip` + 条内再对原分辨率做 tile 滑动**（旧实现）：同样与训练不一致；当前实现已改为「条带整体 resize → 一次推理 → 上采样回原条尺寸」。

## 权重与结构

- **`--weight`**：自行指定 checkpoint（如 machine_split job01）。
- 若 job01 训练时改过 `drop_path_rate` / `mlp_ratio` / `hvss_d_state`，推理需传同名 CLI 覆盖，与 `setting_config` 默认区分。

## `dataset/da/images` 与 `rape_root_seg/img_1`

- **推理输入**：应使用 **`dataset/da/images`** 里与训练一致的预处理图（strip + resize224 管线建立在同一像素分布上）。
- **`img_1`**：同一批素材的**未预处理原图**，可与 DA 图 **同宽高对齐**；若要「给观众看原图叠掩码」，应对 **`img_1` 做可视化**，但 **概率图仍应由 DA 图 forward 得到**（见 `demo_img1_strip_predict_one.py` 的 `--da_image` / `--viz_image`）。

---

*与代码内 `predict_one_strip512` 及 `parse_args` 中 `--mode strip` 说明一致；修改 strip 逻辑时请同步更新本文。*
