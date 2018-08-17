#!/usr/bin/env bash

set -e
set -x

export PYTHONPATH="src"

# run models
for i in `seq 0 12`
do
echo din_ffm_v3_r_$i.yaml start
python src/train_model_din_by_epoch_chunked_r.py --cfg models/din_ffm_v3_r_$i.yaml
python src/train_model_din_by_epoch_chunked_r.py --cfg models/din_ffm_v3_r_$i.yaml --test
echo din_ffm_v3_r_$i.yaml end
done