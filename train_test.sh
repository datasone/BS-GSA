#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

# data dir not required when feature file is supplied, but it's a requied param
mkdir data_not_used
python "$SCRIPTPATH"/main_ca.py -d data_not_used -e 100 -b 512 -s 3197275086 -t 2 -o 1.6 --subset impro --aacnn_mode true --disable_frame_pooling --ee_only --k_fold 5 --gpus 1 --feature_file "$1"
