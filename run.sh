#!/usr/bin/env bash

set -e
set -x

export PYTHONPATH="src"

sh install.sh

# data processing
sh pre-processing.sh

# run models
sh run_models.sh

# result processing
sh post-processing.sh