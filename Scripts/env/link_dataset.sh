#!/usr/bin/env bash
# Link Datasets/<name> into a training code folder as ./dataset
# Usage (from RootExp_V1.0 root):
#   bash Scripts/env/link_dataset.sh Synthetic
#   bash Scripts/env/link_dataset.sh Wheat RootExp/Codes
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NAME="${1:-}"
TARGET_REL="${2:-RootExp/Codes}"
if [[ -z "$NAME" ]]; then
  echo "Usage: $0 <Synthetic|Wheat|PRMI|RPT|BnaRoot> [target_rel]"
  exit 1
fi
SRC="$ROOT/Datasets/$NAME"
DST="$ROOT/$TARGET_REL/dataset"
if [[ ! -d "$SRC" ]]; then
  echo "Missing dataset folder: $SRC"
  echo "Download from Zenodo (https://doi.org/10.5281/zenodo.23983396) into Datasets/$NAME first."
  exit 1
fi
mkdir -p "$(dirname "$DST")"
ln -sfn "$SRC" "$DST"
echo "Linked: $DST -> $SRC"
