# Datasets

Datasets are hosted on Zenodo (not shipped in this GitHub repository):

**https://doi.org/10.5281/zenodo.23983396**

After download, place each dataset here as:

```text
Datasets/
├── Synthetic/{original,npy}/
├── Wheat/{original,npy}/
├── PRMI/{original,npy}/
├── RPT/{original,npy}/
└── BnaRoot/{original,npy}/
```

Zip archives (local backup, outside this repo package):
`../RootExp_Datasets_zenodo_zips/`

Then link into training code:

```bash
bash Scripts/env/link_dataset.sh Synthetic
```
