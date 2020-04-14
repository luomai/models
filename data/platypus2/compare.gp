#!/usr/bin/env gnuplot

set terminal pdf

set output "compare.pdf"
set key right bottom

set xlabel "Seconds"
set ylabel "Evaluation Accuracy"
set yrange [0:1]

plot \
    "1586799542/eval-acc.txt" with linespoints pointtype 4 title 'bs=32', \
    "1586774212/eval-acc.txt" with linespoints pointtype 6 title 'bs=32,64,128', \
    "1586778089/eval-acc.txt" with linespoints pointtype 6 title 'bs=64,128,256'
    # "TODO/eval-acc.txt" with linespoints pointtype 4 title 'bs=64', \
