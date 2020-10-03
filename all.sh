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

# download dataset to $HOME/var/data/cifar
measure ./download-cifar10-data.sh

# generate a init checkpoint
./generate-init.sh

# run static baseline
measure ./train-cifar10-fixed.sh

# run adaptive batch size
measure ./train-cifar10-adaptive.sh
