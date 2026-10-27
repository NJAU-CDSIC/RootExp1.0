#!/bin/bash
# ============================================================
# 云服务器环境配置脚本
# 适用于：UNet, UNet++, DeepLabV3+, SegFormer, SegNet,
#         ITERroot, ESegFormer（7个对比模型）
# 不含 mamba_ssm 等特殊依赖，纯 PyTorch 即可
# ============================================================

# 创建 conda 环境
conda create -n rootseg python=3.8 -y
conda activate rootseg

# PyTorch 1.13 + CUDA 11.7（与你的 vmunet 环境一致）
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# 核心依赖
pip install numpy==1.24.4
pip install scipy==1.10.1
pip install scikit-learn==1.3.2
pip install scikit-image==0.21.0
pip install pillow==10.4.0
pip install matplotlib==3.7.5
pip install tqdm==4.67.1
pip install einops==0.8.1
pip install timm==0.4.12
pip install thop

echo "============================================================"
echo "环境配置完成！"
echo "验证命令: python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
echo "============================================================"
