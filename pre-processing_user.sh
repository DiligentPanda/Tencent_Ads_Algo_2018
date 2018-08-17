#!/usr/bin/env bash
set -x
set -e

export PYTHONPATH="src"

python scripts/construct_mapping.py
python scripts/mapping_features.py
python scripts/count_features.py
