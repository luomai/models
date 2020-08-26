#!/bin/sh
set -e

cd $(dirname $0)

MODEL_PATH=$PWD

export PYTHONWARNINGS='ignore'
export PYTHONPATH=$MODEL_PATH
export TF_CPP_MIN_LOG_LEVEL=3

model_dir=$HOME/tmp/imagenet

if [ -d $model_dir ]; then
    rm -fr $model_dir
fi

FLAGS=
FLAGS="$FLAGS -dd /data/imagenet/records"
FLAGS="$FLAGS -md $model_dir"

H=127.0.0.1:2
port_range=40001-40002

kungfu_run() {
    kungfu-run -q -logdir logs -logfile kungfu-run.log -port 40000 -port-range $port_range -H $H -np $@
}

export CUDA_VISIBLE_DEVICES=2,3

train_imagenet() {
    local epochs=$1
    local np=$2
    local single_bs=$3

    kungfu_run $np \
        python3 \
        official/resnet/imagenet_main.py \
        $FLAGS \
        -bs $((np * single_bs)) \
        -ng $np \
        -te $epochs
}

train_imagenet 1 2 64
