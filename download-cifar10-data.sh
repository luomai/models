#!/bin/sh
set -e

# https://www.cs.toronto.edu/~kriz/cifar.html

DATA_DIR=$HOME/var/data/cifar

download_cifar() {
    local prefix=https://www.cs.toronto.edu/~kriz
    if [ ! -f "$1" ]; then
        curl -sOJ $prefix/$1
    fi
}

mkdir -p $DATA_DIR && cd $DATA_DIR

download_cifar cifar-10-binary.tar.gz
tar -xvf cifar-10-binary.tar.gz
