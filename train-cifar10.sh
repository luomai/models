#!/bin/sh
set -e

now() { date +%s; }

measure() {
    local begin=$(now)
    $@
    local end=$(now)
    local duration=$((end - begin))
    echo "$@ took ${duration}s"
}

cd $(dirname $0)

MODEL_PATH=$PWD

export PYTHONWARNINGS='ignore'
export PYTHONPATH=$MODEL_PATH
export TF_CPP_MIN_LOG_LEVEL=3

model_dir=$HOME/tmp/cifar10

if [ -d $model_dir ]; then
    rm -fr $model_dir
fi

FLAGS=
FLAGS="$FLAGS -md $model_dir"

cap=4
H=127.0.0.1:$cap
port_range=40001-40004

kungfu_run() {
    kungfu-run -q -logdir logs -logfile kungfu-run.log -port 40000 -port-range $port_range -H $H -np $@
}

# export CUDA_VISIBLE_DEVICES=3

train_cifar10() {
    local epochs=$1
    local np=$2
    local single_bs=$3

    kungfu_run $np \
        python3 \
        official/resnet/cifar10_main.py \
        $FLAGS \
        -bs $((np * single_bs)) \
        -ng $np \
        -te $epochs
}

measure train_cifar10 10 4 50
