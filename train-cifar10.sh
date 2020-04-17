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
model_dir=$HOME/tmp/cifar10

if [ -d $model_dir ]; then
    rm -fr $model_dir
fi

cap=4
H=127.0.0.1:$cap
port_range=40001-40004

kungfu_run() {
    kungfu-run -q -logdir logs -logfile kungfu-run.log -port 40000 -port-range $port_range -H $H -np $@
}

# export CUDA_VISIBLE_DEVICES=3

join() {
    local IFS=','
    echo "$*"
}

hooks() {
    echo kungfu_log_step_hook
    echo kungfu_load_init_model_hook
    echo kungfu_save_model_hook
    echo kungfu_consistency_check_hook
    echo kungfu_inspect_graph_hook
}

app_flags() {
    echo -md $model_dir
    echo -dd $data_dir
    echo -hooks $(join $(hooks))
}

train_cifar10() {
    local epochs=$1
    local np=$2
    local single_bs=$3

    kungfu_run $np \
        python3 \
        official/resnet/cifar10_main.py \
        $(app_flags) \
        -bs $((np * single_bs)) \
        -ng $np \
        -te $epochs
}

export START_TIMESTAMP=$(date +%s)

measure train_cifar10 182 4 64
# measure train_cifar10 1 4 64
# measure train_cifar10 1 1 64
# kungfu_run 182 python3 kungfu_experiment/cifar10_main.py
