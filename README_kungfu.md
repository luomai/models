Run KungFu Experiment for OSDI'20 paper (Fig 4)

```bash
# download dataset to $HOME/var/data/cifar
./download-cifar10-data.sh

# run static baseline
./train-cifar10-fixed.sh

# run adaptive batch size
./train-cifar10-adaptive.sh

# Generate the plots
./plot-all.sh
```
