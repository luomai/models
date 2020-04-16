#!/usr/bin/env python3

import sys
import numpy as np


def show_shape(shape):
    return '[%s]' % (', '.join(str(d) for d in shape))


def main(args):
    filename = args[0]
    data = np.load(filename)
    for name in data:
        v = data[name]
        print('%80s %16s    %s' % (name, v.dtype, show_shape(v.shape)))


main(sys.argv[1:])
