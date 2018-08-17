#!/usr/bin/env bash

set -e
set -x

export PYTHONPATH="src"

sh pre-processing_merge.sh
sh pre-processing_user.sh
sh pre-processing_ad.sh
sh pre-processing_his.sh