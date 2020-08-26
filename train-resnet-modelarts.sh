#!/bin/sh
set -e

python3 /home/work/user-job-dir/src/download_data.py

KUNGFU_RUN=/KungFu/bin/kungfu-run
DATA_DIR=/cache/data_dir
SCRIPT=/home/work/user-job-dir/src/official/resnet/imagenet_main.py

export PYTHONWARNINGS='ignore'
export PYTHONPATH=/home/work/user-job-dir/src
export TF_CPP_MIN_LOG_LEVEL=3

CKPT_DIR=/cache/ckpt_dir
mkdir $CKPT_DIR

EPOCHS=120
NUM_GPUS=8
DEVICE_BATCH_SIZE=128

$KUNGFU_RUN -np $NUM_GPUS \
    python3 $SCRIPT \
    -dd $DATA_DIR \
    -md $CKPT_DIR \
    -bs $((NUM_GPUS * DEVICE_BATCH_SIZE)) \
    -ng $NUM_GPUS \
    -te $EPOCHS
