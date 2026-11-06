#!/bin/bash
# ============================================================
# Baseline conda env for paper comparison models:
#   U-Net, U-Net++, DeepLabv3+, SegFormer, SegRoot, ITErRoot,
#   ESegFormer, OCRNet (U-Mamba / RootExp need mamba_ssm separately)
# ============================================================

conda create -n rootseg python=3.8 -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rootseg

pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

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
echo "Environment ready. Activate with: conda activate rootseg"
echo "Verify: python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
echo "============================================================"
