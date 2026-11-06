# -*- coding: utf-8 -*-
"""
Data preparation script for custom dataset
Uses Segmentation/*.txt files for train/val/test split
"""

import numpy as np
import os
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion

# Parameters
height = 512
width = 512
channels = 3

def read_split_file(filepath):
    """读取分割文件，返回文件名列表"""
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def process_image(img_path):
    """处理图像"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((width, height), Image.BILINEAR)
    return np.array(img, dtype=np.float64)

def process_mask(mask_path):
    """
    处理mask，兼容各种格式：
    - 调色板模式(P): 值为0/1或0/255
    - 灰度模式(L): 值为0/1或0/255或其他
    - RGB模式: 自动转灰度
    最终输出: 二值mask，背景0，前景1
    """
    mask = Image.open(mask_path)
    
    # 统一转为灰度数组
    if mask.mode == 'P':
        # 调色板模式：直接读取索引值
        mask_array = np.array(mask, dtype=np.float64)
    elif mask.mode == 'L':
        mask_array = np.array(mask, dtype=np.float64)
    else:
        # RGB或其他模式：转灰度
        mask_array = np.array(mask.convert('L'), dtype=np.float64)
    
    # 自动检测mask值范围并归一化到0-255
    unique_vals = np.unique(mask_array)
    max_val = mask_array.max()
    
    if max_val == 0:
        # 全黑mask，直接返回
        mask_pil = Image.fromarray(np.zeros((height, width), dtype=np.uint8), mode='L')
        return np.array(mask_pil, dtype=np.float64)
    elif max_val <= 1.0:
        # 值是0-1范围，转为0-255
        mask_array = mask_array * 255.0
    elif max_val < 255:
        # 值在1-254之间（如某些数据集用128表示前景）
        # 将非0值统一视为前景
        mask_array = np.where(mask_array > 0, 255.0, 0.0)
    # else: 已经是0-255范围，不处理
    
    # resize（用0-255值）
    mask_pil = Image.fromarray(mask_array.astype(np.uint8), mode='L')
    mask_pil = mask_pil.resize((width, height), Image.LANCZOS)
    mask_array = np.array(mask_pil, dtype=np.float64)
    
    # 二值化：阈值100，保持边缘连续性
    mask_array = np.where(mask_array > 100, 1.0, 0.0)
    
    # 形态学闭运算：连接小断裂
    mask_array = binary_dilation(mask_array, structure=np.ones((2,2)))
    mask_array = binary_erosion(mask_array, structure=np.ones((2,2)))
    
    return mask_array.astype(np.float64)

def process_split(names, split_name):
    """处理一个数据分割"""
    if not names:
        print(f"  {split_name}: 空列表，跳过")
        return None, None
    
    images = np.zeros([len(names), height, width, channels], dtype=np.float64)
    masks = np.zeros([len(names), height, width], dtype=np.float64)
    valid = 0
    
    for idx, name in enumerate(names):
        # 查找图像文件
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            path = os.path.join("images", name + ext)
            if os.path.exists(path):
                img_path = path
                break
        
        # 查找mask文件
        mask_path = os.path.join("masks", name + '.png')
        
        if img_path and os.path.exists(mask_path):
            try:
                images[idx] = process_image(img_path)
                masks[idx] = process_mask(mask_path)
                valid += 1
            except Exception as e:
                print(f"    Error: {name} - {e}")
        else:
            if not img_path:
                print(f"    Missing image: {name}")
            if not os.path.exists(mask_path):
                print(f"    Missing mask: {name}")
    
    print(f"  {split_name}: {valid}/{len(names)} valid pairs")
    return images[:valid], masks[:valid]

############################################################# Main #################################################
print("Reading Segmentation split files...")
train_names = read_split_file("Segmentation/train.txt")
val_names = read_split_file("Segmentation/val.txt")
test_names = read_split_file("Segmentation/test.txt")

print(f"  train.txt: {len(train_names)} samples")
print(f"  val.txt: {len(val_names)} samples")
print(f"  test.txt: {len(test_names)} samples")

print("\nProcessing datasets...")
Train_img, Train_mask = process_split(train_names, "Train")
Val_img, Val_mask = process_split(val_names, "Val")
Test_img, Test_mask = process_split(test_names, "Test")

print("\nSaving npy files...")
if Train_img is not None and len(Train_img) > 0:
    np.save('data_train.npy', Train_img)
    np.save('mask_train.npy', Train_mask)
    print(f"  data_train.npy, mask_train.npy ({len(Train_img)} samples)")

if Val_img is not None and len(Val_img) > 0:
    np.save('data_val.npy', Val_img)
    np.save('mask_val.npy', Val_mask)
    print(f"  data_val.npy, mask_val.npy ({len(Val_img)} samples)")

if Test_img is not None and len(Test_img) > 0:
    np.save('data_test.npy', Test_img)
    np.save('mask_test.npy', Test_mask)
    print(f"  data_test.npy, mask_test.npy ({len(Test_img)} samples)")
else:
    # 生成空的test文件，避免train.py报错
    np.save('data_test.npy', np.zeros([1, height, width, channels], dtype=np.float64))
    np.save('mask_test.npy', np.zeros([1, height, width], dtype=np.float64))
    print(f"  data_test.npy, mask_test.npy (empty placeholder)")

print("\nDone!")
