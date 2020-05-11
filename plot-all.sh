#!/bin/sh
set -e

plot() {
    local prefix=$1
    gnuplot -c plot.gp \
        ${prefix}/plot.pdf \
        ${prefix}/train-loss.txt \
        ${prefix}/train-acc.txt \
        ${prefix}/eval-loss.txt \
        ${prefix}/eval-acc.txt
}

model_dir_prefix=$HOME/tmp/cifar10

plot_job() {
    local job_name=$1
    local log_dir=$model_dir_prefix/$job_name/0

    echo "extract events for $job_name from logs in $log_dir"
    local dir=$(./events.py $log_dir $job_name)

    echo "plot for $job_name from data/$dir"
    plot data/$dir
}

# plot_job fixed-bs-32
# plot_job fixed-bs-64
# plot_job fixed-bs-128
# plot_job fixed-bs-256
# plot_job fixed-bs-512
# plot_job fixed-bs-1024

plot_job adaptive-bs-32
