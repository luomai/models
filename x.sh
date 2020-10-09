#!/bin/sh
set -e

notify() {
    if [ -f ~/.slack/notify ]; then
        ~/.slack/notify "$@"
    fi
}

now() { date +%s; }

measure() {
    local begin=$(now)
    $@
    local end=$(now)
    local duration=$((end - begin))
    echo "$@ took ${duration}s"
}

main() {

    # download dataset to $HOME/var/data/cifar
    measure ./download-cifar10-data.sh

    # generate a init checkpoint
    measure ./generate-init.sh

    # run adaptive batch size
    measure ./train-cifar10-adaptive.sh

    # run static baseline
    measure ./train-cifar10-fixed.sh

}

notify "BEGIN $0"
measure main
notify "END $0"
