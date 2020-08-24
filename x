#!/bin/sh
set -e

# ./train-cifar10.sh

nohup_run() {
    $@ >out.log 2>err.log &
}

nohup_run ./train-cifar10-adaptive.sh
