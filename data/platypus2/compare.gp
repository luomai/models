#!/usr/bin/env gnuplot

set terminal pdf

set key right bottom

set xlabel "Seconds"
set ylabel "Evaluation Accuracy"
set yrange [0:1]

# set output "compare-32.pdf"
# plot \
#     "1586799542/eval-acc.txt" with linespoints pointtype 4 title 'bs=4x32', \
#     "1586774212/eval-acc.txt" with linespoints pointtype 6 title 'bs=4x{32,64,128}'

# set output "compare-64.pdf"
# plot \
#     "1586871328/eval-acc.txt" with linespoints pointtype 4 title 'bs=4x64', \
#     "1586778089/eval-acc.txt" with linespoints pointtype 6 title 'bs=4x{64,128,256}'

# set output "compare-64-2.pdf"
# plot \
#     "1586880326/eval-acc.txt" with linespoints pointtype 4 title 'bs=4x64', \
#     "1586882990/eval-acc.txt" with linespoints pointtype 6 title 'bs=4x{64,128,256}'


# set output "compare-64-3.pdf"
# plot \
#     "1586888710/eval-acc.txt" with linespoints pointtype 4 title 'bs=4x64', \
#     "1586886256/eval-acc.txt" with linespoints pointtype 6 title 'bs=4x{64,128,256}'

set output "compare-fixed-bs.pdf"
plot \
    "fixed-bs-32/eval-acc.txt"  with linespoints title 'bs=4x32',  \
    "fixed-bs-64/eval-acc.txt"  with linespoints title 'bs=4x64',  \
    "fixed-bs-128/eval-acc.txt" with linespoints title 'bs=4x128', \
    "fixed-bs-256/eval-acc.txt" with linespoints title 'bs=4x256', \
    "fixed-bs-512/eval-acc.txt" with linespoints title 'bs=4x512', \
