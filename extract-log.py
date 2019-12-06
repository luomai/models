#!/usr/bin/env python3

import json
import sys


def grep(filename, pattern):
    for line in open(filename):
        if pattern in line:
            yield line.strip()


def extract(lines):
    ds = dict()
    for line in lines:
        line = line.strip()
        mark = 'Benchmark metric:'
        p = line.find(mark)
        if p < 0:
            continue
        line = line[p + len(mark):]
        line = line.replace('\'', '"').replace('None', 'null')
        record = json.loads(line)

        key = record['name']
        gs = int(record['global_step'])
        # ts = record['timestamp']
        val = float(record['value'])

        ds.setdefault(gs, dict())[key] = val
    return ds


def show(ds):
    keys = ['loss', 'accuracy', 'accuracy_top_5']
    xs = sorted(ds.keys())
    for x in xs:
        loss, acc, top5acc = [ds[x][k] for k in keys]
        print('%d %f %f %f' % (x, loss, acc, top5acc))


def main(args):
    if len(args) != 2:
        print('Usage:\n\t%s <filename>' % (args[0]))
        return
    filename = args[1]
    ds = extract(grep(filename, '127.0.0.1.10000'))
    show(ds)


main(sys.argv)
