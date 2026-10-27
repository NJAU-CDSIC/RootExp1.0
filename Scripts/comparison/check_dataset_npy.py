#!/usr/bin/env python3
"""检查 data/mask npy 是否与 unet/loader.py 逻辑兼容（无 NaN/Inf、mask 范围、归一化是否除零）。"""
import os
import sys
import numpy as np

# 与 model/unet/loader.py 一致
def dataset_normalized(imgs):
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    if imgs_std == 0 or not np.isfinite(imgs_std):
        return None, f"全局 std 异常: std={imgs_std}"
    imgs_normalized = (imgs - imgs_mean) / imgs_std
    bad_idx = []
    for i in range(imgs.shape[0]):
        lo, hi = np.min(imgs_normalized[i]), np.max(imgs_normalized[i])
        if hi - lo == 0 or not np.isfinite(hi - lo):
            bad_idx.append((i, float(lo), float(hi)))
        else:
            imgs_normalized[i] = ((imgs_normalized[i] - lo) / (hi - lo)) * 255.0
    if bad_idx:
        return None, f"常数图(无法 min-max): 共 {len(bad_idx)} 张, 示例 index={bad_idx[:5]}"
    if not np.isfinite(imgs_normalized).all():
        return None, "归一化后出现非有限值"
    return imgs_normalized, None


def count_constant_images_after_zscore(arr):
    """z-score 后每张图若 min==max，第二步 min-max 会除零（与 loader 一致）。"""
    imgs_std = np.std(arr)
    imgs_mean = np.mean(arr)
    if imgs_std == 0 or not np.isfinite(imgs_std):
        return -1, []
    z = (arr - imgs_mean) / imgs_std
    bad = []
    for i in range(arr.shape[0]):
        lo, hi = float(np.min(z[i])), float(np.max(z[i]))
        if hi - lo == 0 or not np.isfinite(hi - lo):
            bad.append(i)
    return len(bad), bad[:10]


def mask_after_loader_logic(mask):
    m = np.expand_dims(mask, axis=3)
    if m.max() <= 1.0:
        pass
    else:
        m = m / 255.0
    return m


def check_one(name, base_dir):
    print(f"\n{'='*60}")
    print(f"数据集: {name}")
    print(f"路径: {base_dir}")
    files = ["data_train", "mask_train", "data_val", "mask_val", "data_test", "mask_test"]
    for stem in files:
        p = os.path.join(base_dir, f"{stem}.npy")
        if not os.path.isfile(p):
            continue
        data = np.load(p, mmap_mode="r")
        kind = "data" if stem.startswith("data") else "mask"
        print(f"\n--- {stem}.npy  shape={data.shape}  dtype={data.dtype}")
        if kind == "mask":
            mx, mn = float(np.max(data)), float(np.min(data))
            print(f"    mask 原始 min={mn:.6g}  max={mx:.6g}")
            if not np.isfinite(data).all():
                n_bad = np.size(data) - np.isfinite(data).sum()
                print(f"    [问题] 非有限值像素数: {n_bad}")
            m2 = mask_after_loader_logic(np.array(data))
            print(f"    loader 后 min={m2.min():.6g} max={m2.max():.6g}")
            # 抽样 unique（大图只抽）
            flat = np.array(data).ravel()
            if flat.size > 200_000:
                flat = flat[np.random.RandomState(0).choice(flat.size, 200_000, replace=False)]
            u = np.unique(flat)
            if u.size <= 20:
                print(f"    unique(抽样或全量): {u}")
            else:
                print(f"    unique 数量: {u.size}  (前10) {u[:10]}")
        else:
            arr = np.array(data)
            if not np.isfinite(arr).all():
                print(f"    [问题] 原始 data 含非有限值")
            n_bad, sample_idx = count_constant_images_after_zscore(arr)
            if n_bad > 0:
                print(f"    [警告] z-score 后为常数图(第二步将除零): {n_bad} 张, 示例 index={sample_idx}")
            elif n_bad == -1:
                print(f"    [问题] 无法做 z-score（std=0）")
            out, err = dataset_normalized(arr)
            if err:
                print(f"    [问题] dataset_normalized: {err}")
            else:
                print(f"    归一化后 min={out.min():.4f} max={out.max():.4f}  isfinite={np.isfinite(out).all()}")


def main():
    roots = [
        "/home/young/桌面/testGeModel/dataset/RPT",
        "/home/young/桌面/testGeModel/dataset/prim",
        "/home/young/桌面/testGeModel/dataset/sumilte",
        "/home/young/桌面/testGeModel/dataset/wheat",
        "/home/young/桌面/testGeModel/dataset/da/da_cropped",
    ]
    for d in roots:
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "data_train.npy")):
            check_one(os.path.basename(os.path.normpath(d)), d)
        else:
            print(f"\n跳过(不存在或无 data_train.npy): {d}")

    if len(sys.argv) > 1:
        extra = sys.argv[1]
        if os.path.isdir(extra):
            check_one(os.path.basename(os.path.normpath(extra)), extra)


if __name__ == "__main__":
    main()
