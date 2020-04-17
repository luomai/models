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
model_dir=$HOME/tmp/cifar10/init

if [ -d $model_dir ]; then
    rm -fr $model_dir
fi

join() {
    local IFS=','
    echo "$*"
}

hooks() {
    echo kungfu_save_init_model_hook
    # echo kungfu_load_init_model_hook
}

app_flags() {
    echo -md $model_dir
    # echo -dd $data_dir/cifar-10-batches-bin
    echo -dd $data_dir
    echo -hooks $(join $(hooks))
    echo -te 1
}

main() {
    python3 \
        official/resnet/cifar10_main.py \
        $(app_flags)
}

measure main
