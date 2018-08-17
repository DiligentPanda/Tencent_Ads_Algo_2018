#!/usr/bin/env bash
set -x
set -e

export PYTHONPATH="src"

python scripts/construct_mapping_ad.py
python scripts/mapping_features_ad.py
python scripts/count_features_ad.py
