#!/usr/bin/env bash
set -x
set -e

export PYTHONPATH="src"

mkdir data/M_ff
python scripts/merge/merge_ads.py
python scripts/merge/merge_train.py
python scripts/merge/merge_users.py
cp data/F/test2.csv data/M_ff/test2.csv