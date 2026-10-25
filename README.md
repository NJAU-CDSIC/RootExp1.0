# **RootExp** Code Repository

This is a directory for storing the **RootExp** model code and data. **RootExp** (network: `RootBin_Block_Mamba_EAM_ASPP`) is a deep learning framework for **binary plant-root image segmentation**, combining a U-Net-style encoder–decoder with ML-Mamba (HVSS), EAM, and ASPP modules.

---

The folders in the RootExp repository:

- **Datasets (figshare)**:

   a. **Synthetic (sumilte)**: Synthetic root images and masks (original RGB/masks + preprocessed `.npy`).

   b. **Wheat**: Wheat root phenotyping images and masks (original + `.npy`).

   c. **PRMI (prim)**: PRMI root images and masks (original + `.npy`).

   d. **RPT**: Root Phenotyping Toolbox images and masks (original + `.npy`).

   e. **BnaRoot (da)**: Oilseed-rape / BnaRoot field strips (original full images + cropped `.npy` patches).

   Each dataset folder contains `original/` (raw `images/` + `masks/`) and `npy/` (`data_*.npy` / `mask_*.npy`).  
   A single upload archive is available at `Datasets (figshare)/_figshare_zips/RootExp_Datasets_all.zip`.  
   Figshare: https://doi.org/10.6084/m9.figshare.34776657

- **RootExp_model**:

   a. **RootExp_code**: Main training / inference code for RootExp (`train.py`, `predict_image.py`, `models/rootbin.py`).

   b. **Pre-trained**: Released checkpoints, including RootExp (main) and RootExp-DA (strip / BnaRoot), plus optional ONNX.

- **SOTA**: Comparative methods used in the contrast experiments :

   U-Net: Ronneberger et al., MICCAI 2015

   U-Net++: Zhou et al., DLMIA 2018

   DeepLabv3+: Chen et al., 2018

   SegFormer: Xie et al., NeurIPS 2021

   U-Mamba: Ma et al. (U-Mamba / Mamba-based U-Net)

   SegRoot: Wang et al. (root-specific)

   ITErRoot: Seitz et al., 2022 (root-specific)

   ESegFormer: Shi et al., 2025 (SegFormer + EAM)

   OCRNet: Huang et al., Front. Plant Sci. 2023 (improved OCRNet for cotton-root minirhizotron images; domain-specific)

- **Scripts**: Contains auxiliary scripts for inference, comparison, case study, and environment setup.

   a. **inference**: DA / panorama strip inference helpers (CLAHE preprocess, strip predict, ONNX export).

   b. **comparison**: Multi-baseline evaluation helpers (resize-224 / strip protocols).

   c. **case_study**: Qualitative case-study assembly and figure scripts.

   d. **env**: `setup_env.sh`, `link_dataset.sh`.

---

### **Step-by-step Running:**

## 1. Environment Installation

It is recommended to use a conda environment. Two setups are used in this project:

- **Baselines / general tools** (Python 3.8, PyTorch 1.13 + CUDA 11.7): see `Scripts/env/setup_env.sh`.

- **RootExp** (PyTorch + `mamba_ssm` / VMamba stack): use the same environment as your RootExp training machine (e.g. `vmunet`).

Install the baseline runtime environment:

```bash
bash Scripts/env/setup_env.sh
conda activate rootseg
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Datasets

Place the datasets under `Datasets (figshare)/`, or download from Figshare: https://doi.org/10.6084/m9.figshare.34776657

Expected layout per dataset:

```text
Datasets (figshare)/<name>/
├── original/          # images/ + masks/ (+ Segmentation/)
└── npy/               # data_{train,val,test}.npy , mask_{train,val,test}.npy
```

Link a dataset into the RootExp training path:

```bash
bash Scripts/env/link_dataset.sh sumilte   # or wheat | prim | RPT | da
```

Optional: rebuild `.npy` from original images with `Datasets (figshare)/Prepare_your_dataset.py`.

## 3. Training and Testing

After setting up the environment and datasets, train / run RootExp from `RootExp_model/RootExp_code/`.

- Train the full-width model:

```bash
cd RootExp_model/RootExp_code
python train.py
```



- Inference with a released checkpoint (example):

```bash
cd RootExp_model/RootExp_code
python predict_image.py \
  --weight ../Pre-trained/RootExp_best-epoch1927-loss0.1049.pth \
  --image /path/to/image.png
```

Make sure the `Pre-trained/` folder contains the corresponding `.pth` before testing.

## 4. Scripts Usage

- **DA strip / panorama inference** (large field images):

```bash
cd Scripts/inference
python predict_da_strip_one.py --help
```

- **Case-study assembly**:

```bash
cd Scripts/case_study
# see README.md for figure / mask assembly commands
```

- **SOTA model code** 

```bash
ls SOTA/models/
# U-Net  U-Net++  DeepLabv3+  SegFormer  U-Mamba  SegRoot  ITErRoot  ESegFormer  OCRNet
```

## 5. Installation

Download the code:

```bash
git clone https://github.com/<YOUR_ORG_OR_USER>/RootExp.git
cd RootExp
```

## 6. Citation

If you use RootExp in your research, please cite:

> Datasets: https://doi.org/10.6084/m9.figshare.34776657
