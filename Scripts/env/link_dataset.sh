#!/usr/bin/env bash
# Link Datasets (figshare)/<name> into a training code folder as ./dataset
# Usage (from RootExp_V1.0 root):
#   bash Scripts/env/link_dataset.sh sumilte
#   bash Scripts/env/link_dataset.sh wheat RootExp_model/RootExp_code
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NAME="${1:-}"
TARGET_REL="${2:-RootExp_model/RootExp_code}"
if [[ -z "$NAME" ]]; then
  echo "Usage: $0 <sumilte|wheat|prim|RPT|da> [target_rel]"
  exit 1
fi
SRC="$ROOT/Datasets (figshare)/$NAME"
DST="$ROOT/$TARGET_REL/dataset"
if [[ ! -d "$SRC" ]]; then
  echo "Missing dataset folder: $SRC"
  echo "Download from Figshare into Datasets (figshare)/$NAME first."
  exit 1
fi
mkdir -p "$(dirname "$DST")"
ln -sfn "$SRC" "$DST"
echo "Linked: $DST -> $SRC"
