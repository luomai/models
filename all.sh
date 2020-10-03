#!/bin/sh
set -e

# download dataset to $HOME/var/data/cifar
./download-cifar10-data.sh

# generate a init checkpoint
./generate-init.sh

# run static baseline
./train-cifar10-fixed.sh

# run adaptive batch size
./train-cifar10-adaptive.sh
