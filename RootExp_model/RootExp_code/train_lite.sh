#!/usr/bin/env bash
# Train RootExp-Lite (narrower channels) using the same train.py entry.
# Usage: from this directory
#   bash train_lite.sh
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-/home/young/anaconda3/envs/vmunet/bin/python}"

# Point setting_config at the lite preset for this run (train.py imports configs.config_setting).
cp -f configs/config_setting.py configs/config_setting_full_backup.py
cp -f configs/config_setting_lite.py configs/config_setting.py
# Keep network name distinguishable in work_dir; restore class fields from lite file.
trap 'cp -f configs/config_setting_full_backup.py configs/config_setting.py; echo "restored full config_setting.py"' EXIT

echo "Training RootExp-Lite with encoder_channels=[36,72,144,288,384]"
"$PY" train.py
