#!/usr/bin/env bash
set -x
set -e

export PYTHONPATH="src"

python scripts/user_history_4.py
# note the constant in file gen_feature_info_r_2.py, they can be read from the last print line in user_history_4.py
# 1005 is the number of advertisement
# 2951 is the number of different feature values (of all fields).
python scripts/gen_feature_info_r_2.py