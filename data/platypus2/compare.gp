#!/usr/bin/env gnuplot

set terminal pdf

BASE_LINE_EVAL_ACC = "1586799542/eval-acc.txt"
KUNGFU_EVAL_ACC = "1586774212/eval-acc.txt"

set output "compare.pdf"
set key right bottom

set xlabel "Seconds"
set ylabel "Evaluation Accuracy"

plot \
    BASE_LINE_EVAL_ACC with linespoints pointtype 4 title 'bs=32', \
    KUNGFU_EVAL_ACC with linespoints pointtype 6 title 'bs=32,64,128'
