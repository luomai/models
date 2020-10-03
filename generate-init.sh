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

data_dir=$HOME/var/data/cifar
model_dir_prefix=$HOME/tmp/cifar10
job_id=init
model_dir=$model_dir_prefix/$job_id

export CUDA_VISIBLE_DEVICES=0

join() {
    local IFS=','
    echo "$*"
}

hooks() {
    echo kungfu_log_step_hook
    echo kungfu_save_init_model_hook
}

app_flags() {
    echo -md $model_dir
    echo -dd $data_dir
    echo -hooks $(join $(hooks))
    echo -kungfu_opt gns
}

main() {
    python3 \
        official/resnet/cifar10_main.py \
        $(app_flags) \
        -bs 32 \
        -ng 1 \
        -te 1
}

measure main
