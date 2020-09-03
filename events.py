#!/usr/bin/env python3

import os
import sys
import glob

import tensorflow as tf
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False


def select_tags(idx, event, tags=[]):
    if event.step == 0:
        return
    value_map = dict()
    cnt = 0
    for value in event.summary.value:
        if value.tag in tags:
            value_map[value.tag] = value.simple_value
            cnt += 1
    if cnt != len(tags):
        return

    values = [value_map[t] for t in tags]
    columns = [
        # idx,
        # event.step,
        event.wall_time,
    ]
    for v in values:
        columns.append(v)
    return columns


def get_all_tags(filename):
    tags = dict()
    for event in tf.train.summary_iterator(filename):
        for value in event.summary.value:
            tags[value.tag] = True
    return tags.keys()


def save_columns(prefix, name, *columns):
    filename = os.path.join(prefix, name)
    with open(filename, 'w') as f:
        rows = zip(*columns)
        for row in rows:
            f.write(' '.join(str(x) for x in row) + '\n')


def select_events(filename, select_tags, tags, t0):
    rows = []
    for idx, event in enumerate(tf.train.summary_iterator(filename)):
        row = select_tags(idx, event, tags)
        if row:
            rows.append(row)

    wall_time_idx = 0
    for row in rows:
        row[wall_time_idx] -= t0

    return rows


def find_events(folder):
    pattern = os.path.join(folder, 'events.out.tfevents.*')
    return [f for f in glob.glob(pattern)]


def process_folder(ckpt_dir):
    filenames = find_events(ckpt_dir)
    # print(filenames)
    filename = os.path.join(ckpt_dir, filenames[0])
    _, _, _, t0, host = os.path.basename(filename).split('.')
    # print(get_all_tags(filename))
    return host, int(t0), filename


def main(ckpt_dir, job_name=None):
    train_tags = [
        'loss',
        'train_accuracy_1',
    ]
    eval_tags = [
        'loss',
        'accuracy',
    ]

    output_dir = 'data'

    host, t0, filename = process_folder(ckpt_dir)
    rows = select_events(filename, select_tags, train_tags, t0)

    ts, loss, acc = zip(*rows)
    if job_name is None:
        job_name = str(t0)
    prefix = os.path.join(output_dir, host, job_name)

    os.makedirs(prefix, exist_ok=True)
    save_columns(prefix, 'train-loss.txt', ts, loss)
    save_columns(prefix, 'train-acc.txt', ts, acc)

    _, _, filename = process_folder(os.path.join(ckpt_dir, 'eval'))
    rows = select_events(filename, select_tags, eval_tags, t0)

    ts, loss, acc = zip(*rows)
    save_columns(prefix, 'eval-loss.txt', ts, loss)
    save_columns(prefix, 'eval-acc.txt', ts, acc)

    print('%s/%s' % (host, job_name))


job_name = None
if len(sys.argv) > 2:
    job_name = sys.argv[2]
main(sys.argv[1], job_name)
