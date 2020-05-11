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

data_dir=$HOME/tmp/data
model_dir_prefix=$HOME/tmp/cifar10
model_dir=$model_dir_prefix


cap=4
H=127.0.0.1:$cap
port_range=40001-40004

kungfu_run_flags() {
    echo -q
    echo -logdir logs/$job_id
    echo -logfile kungfu-run.log
    echo -port 40000
    echo -port-range $port_range
    echo -H $H
    echo -np
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
}

# export CUDA_VISIBLE_DEVICES=3

join() {
    local IFS=','
    echo "$*"
}

hooks() {
    echo kungfu_log_step_hook

    # echo kungfu_load_init_model_hook
    # echo kungfu_save_model_hook
    # echo kungfu_consistency_check_hook
    # echo kungfu_inspect_graph_hook

    # echo kungfu_change_batch_size_hook
}

app_flags() {
    echo -md $model_dir
    echo -dd $data_dir
    echo -hooks $(join $(hooks))
    echo -kungfu_opt ssgd
    # echo -kungfu_opt gns
}

train_cifar10() {
    local epochs=$1
    local np=$2
    local single_bs=$3

    export START_TIMESTAMP=$(date +%s)
    job_id=fixed-bs-$single_bs

    model_dir=$model_dir_prefix/$job_id

    if [ -d $model_dir ]; then
        rm -fr $model_dir
    fi

    kungfu_run $np \
        python3 \
        official/resnet/cifar10_main.py \
        $(app_flags) \
        -bs $single_bs \
        -ng $np \
        -te $epochs
}

run_all() {
    local epochs=300
    #local epochs=10
    #local epochs=1

    # measure train_cifar10 $epochs 4 32
    # measure train_cifar10 $epochs 4 64
    # measure train_cifar10 $epochs 4 128
    # measure train_cifar10 $epochs 4 256
    # measure train_cifar10 $epochs 4 512

    measure train_cifar10 $epochs 4 1024
}

measure run_all
