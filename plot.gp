#!/usr/bin/env gnuplot -c

set terminal pdf

OUT  = ARG1
LOSS = ARG2
ACC  = ARG3
EVAL_LOSS = ARG4
EVAL_ACC = ARG5

set output OUT

plot \
    LOSS with lines title 'loss', \
    ACC with lines title 'acc', \
    EVAL_LOSS with lines title 'eval-loss', \
    EVAL_ACC with lines title 'eval-acc'
