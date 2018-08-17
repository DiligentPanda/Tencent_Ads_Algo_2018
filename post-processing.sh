#!/usr/bin/env bash
set -x
set -e

export PYTHONPATH="src"

python scripts/collect_res.py
python scripts/gen_file_list.py
python scripts/score_avg.py