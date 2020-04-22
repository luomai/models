#!/usr/bin/env gnuplot

set terminal pdf

IN  = ARG1
OUT = ARG2

set output OUT

plot \
    IN with lines title 'gns'
