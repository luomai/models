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

main() {
    local output=$1
    dir=$(./events.py $output)
    plot data/$dir
}

main $HOME/tmp/cifar10/0
