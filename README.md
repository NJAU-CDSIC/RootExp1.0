# RootExp: a structure-aware segmentation framework for plant root analysis

This repository contains the complete implementation of **RootExp**, a structure-aware deep learning framework for **binary plant-root image segmentation**. 

---

The folders in the RootExp repository:

- **RootExp**: Source code and released weights for RootExp

  - **Codes**: Training / inference entry points and model definition

    - `models/`: Core network (`rootbin.py`; ML-Mamba and related modules)

    - `configs/`: Training config (`config_setting.py`)

    - `train.py`: Train RootExp

    - `predict_image.py`: Single-image inference

    - `loader.py`, `engine.py`, `utils.py`: Data loading and training helpers

  - **Pre-trained**: Released checkpoint `best.pt` (optional ONNX). 

- **Datasets**: Root segmentation datasets (original RGB/masks + preprocessed `.npy`)

  - Five datasets used in the paper:

    - **Synthetic**: Synthetic root images and masks

    - **Wheat**: Wheat root phenotyping images and masks

    - **PRMI**: PRMI root images and masks

    - **RPT**: Root Phenotyping Toolbox images and masks

    - **BnaRoot**: Oilseed-rape / BnaRoot field panoramas (full strips + cropped patches)

  - Each dataset folder contains `original/` (`images/` + `masks/`) and `npy/` (`data_*.npy` / `mask_*.npy`)

  - `Prepare_your_dataset.py`: Optional rebuild of `.npy` from original images

  - Full archives on Zenodo <https://doi.org/10.5281/zenodo.23983396>:

    ```
    Synthetic_original.zip   Synthetic_npy.zip
    Wheat_original.zip       Wheat_npy.zip
    PRMI_original.zip        PRMI_npy.zip
    RPT_original_part{1,2,3}_of_3.zip   RPT_npy.zip
    BnaRoot_original.zip     BnaRoot_npy.zip
    ```

    (RPT original is split into three parts for upload size limits. Local zip backup: `../RootExp_Datasets_zenodo_zips/`.)

- **SOTA Models**: Nine models used in the comparative study (**architecture code only**):

  - U-Net: Ronneberger et al., MICCAI 2015

  - U-Net++: Zhou et al., DLMIA 2018

  - DeepLabv3+: Chen et al., 2018

  - SegFormer: Xie et al., NeurIPS 2021

  - U-Mamba: Ma et al. (U-Mamba / Mamba-based U-Net)

  - SegRoot: Wang et al. (root-specific)

  - ITErRoot: Seitz et al., 2022 (root-specific)

  - ESegFormer: Shi et al., 2025 (SegFormer + EAM)

  - OCRNet: Huang et al., Front. Plant Sci. 2023 — improved OCRNet for cotton-root minirhizotron images (domain-specific; DOI: [10.3389/fpls.2023.1147034](https://doi.org/10.3389/fpls.2023.1147034))

- **Scripts**: Auxiliary scripts for environment, inference, and timing benchmarks

  - **env**: `setup_env.sh` (baseline conda env), `link_dataset.sh` (link a dataset into the training path)

  - **inference**: RootExp / BnaRoot strip inference ( strip predict, ONNX export)

  - **comparison**: `benchmark_inference_ms.py` (GPU inference latency @ 224×224)

---

## **Step-by-step Running:**

## 1. Installation

```bash
git clone https://github.com/NJAU-CDSIC/RootExp1.0.git
cd RootExp
```

## 2. Environment Installation

It is recommended to use conda. Two setups are used in this project:

- **Baselines / general tools** (Python 3.8, PyTorch 1.13 + CUDA 11.7):

```bash
bash Scripts/env/setup_env.sh
conda activate rootseg
```

- **RootExp**: use the same conda environment as your RootExp training machine (e.g. `vmunet`; CUDA + selective-scan deps required for ML-Mamba).

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 3. Datasets and model parameters

To keep the GitHub repository within platform size limits, **datasets** (and optionally large checkpoints) are hosted on Zenodo. **Full functionality requires downloading these resources separately**:

### Essential Zenodo Resources

| Resource Category | Zenodo Access | Required For |
|:-----------------:|:-------------:|:------------:|
| **Datasets** (original + `.npy`) | <https://doi.org/10.5281/zenodo.23983396> | Training, evaluation, case studies |


Expected layout after download:

```text
Datasets/<name>/
├── original/          # images/ + masks/ (+ Segmentation/)
└── npy/               # data_{train,val,test}.npy , mask_{train,val,test}.npy
```

where `<name>` is one of: `Synthetic` | `Wheat` | `PRMI` | `RPT` | `BnaRoot`.

Link a dataset into the RootExp training path:

```bash
bash Scripts/env/link_dataset.sh Synthetic   # or Wheat | PRMI | RPT | BnaRoot
```

## 4. Training and inference

Train RootExp:

```bash
cd RootExp/Codes
python train.py
```

Inference with a released checkpoint (example):

```bash
cd RootExp/Codes
python predict_image.py \
  --weight ../Pre-trained/best.pt \
  --image /path/to/image.png
```

BnaRoot strip / panorama inference (large field images):

```bash
cd Scripts/inference
python predict_bnaroot_strip.py --help
```

SOTA model code (definitions only):

```bash
ls "SOTA Models/"
# U-Net  U-Net++  DeepLabv3+  SegFormer  U-Mamba  SegRoot  ITErRoot  ESegFormer  OCRNet
```

## 5. Citation

If you use RootExp in your research, please cite this repository and the dataset

- Code: `https://github.com/NJAU-CDSIC/RootExp1.0.git`
- Datasets: <https://doi.org/10.5281/zenodo.23983396>


