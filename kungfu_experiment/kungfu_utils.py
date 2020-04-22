import os
import time

from absl import flags
from kungfu import current_rank

import numpy as np
import tensorflow as tf
from official.utils.flags import core as flags_core

USE_DYNAMIC_BATCH_SIZE = False
# USE_DYNAMIC_BATCH_SIZE = True

START_TIMESTAMP = os.getenv('START_TIMESTAMP')
KUNGFU_OPT = None  # parsed from flags
# KUNGFU_OPT = 'ssgd'
# KUNGFU_OPT = 'gns'


def define_kungfu_flags():
    flags.DEFINE_string(name='kungfu_opt',
                        default='ssgd',
                        help=flags_core.help_wrap('ssgd | gns'))


class KungfuLogStepHook(tf.train.SessionRunHook):
    def __init__(self, **kwargs):
        self._global_step = 0
        self._iter_step = 0

    def begin(self):
        self._t_begin = time.time()
        self._iter_step = 0
        print('%s::%s(%d)' %
              (self.__class__.__name__, 'begin', self._global_step))

    def after_create_session(self, sess, coord):
        dur = time.time() - self._t_begin
        print('%s::%s(%d), took %.3f since begin' %
              (self.__class__.__name__, 'after_create_session',
               self._global_step, dur))

    def before_run(self, run_context):
        self._t_before = time.time()
        if self._iter_step % 20 == 0:
            print('%s::%s(%d, %d)' % (self.__class__.__name__, 'before_run',
                                      self._iter_step, self._global_step))

    def after_run(self, run_context, run_values):
        dur = time.time() - self._t_before
        if self._iter_step % 20 == 0:
            print('%s::%s(%d, %d), %.3fs per step' %
                  (self.__class__.__name__, 'after_run', self._iter_step,
                   self._global_step, dur))
        self._global_step += 1
        self._iter_step += 1

    def end(self, sess):
        dur = time.time() - self._t_begin
        print('%s::%s(%d), took %.2fs' %
              (self.__class__.__name__, 'end', self._global_step, dur))


_MODEL_FILE = 'init.npz'


def save_model(filename, sess, variables):
    values = dict()
    vs = sess.run(variables)
    for v, value in zip(variables, vs):
        values[v.name] = value
    np.savez(filename, **values)


def load_model(filename, sess, variables, placeholders, assign_ops):
    data = np.load(filename)
    values = dict()
    for name in data:
        values[name] = data[name]
    feed_dict = dict()
    for v, p in zip(variables, placeholders):
        feed_dict[p] = values[v.name]
    sess.run(assign_ops, feed_dict=feed_dict)


class KungfuSaveInitModelHook(tf.train.SessionRunHook):
    def __init__(self, **kwargs):
        print('%s created' % (self.__class__.__name__))

    def begin(self):
        self._variables = tf.global_variables()
        print('model has %d variables' % (len(self._variables)))

    def after_create_session(self, sess, coord):
        t0 = time.time()
        save_model(_MODEL_FILE, sess, self._variables)
        print('model saved to %s, took %.2fs' %
              (_MODEL_FILE, time.time() - t0))

    def before_run(self, run_context):
        run_context.request_stop()


def diff_list(a, b):
    c = set(a)
    d = set(b)
    return list(c - d), list(d - c)


def checkpoint_name(prefix, cycle, step):
    rank = current_rank()
    return 'job-%s-%s-cycle-%03d-step-%08d-rank-%02d' % (
        START_TIMESTAMP, prefix, cycle, step, rank)


class KungfuSaveModelHook(tf.train.SessionRunHook):
    def __init__(self, **kwargs):
        print('%s created' % (self.__class__.__name__))
        self._step = 0
        self._cycle = 0
        self._names = []

    def begin(self):
        # if self._cycle == 0:
        self._update_variable_list()

    def after_run(self, run_context, run_values):
        self._step += 1
        if self._step <= 2:
            self._save('after_run', run_context.session)

    def end(self, sess):
        self._save('end', sess)
        self._cycle += 1

    def _save(self, prefix, sess):
        filename = checkpoint_name(prefix, self._cycle, self._step)
        save_model(filename, sess, self._variables)

    def _update_variable_list(self):
        self._variables = tf.global_variables()
        names = [v.name for v in self._variables]
        if self._names:
            c, d = diff_list(self._names, names)
            if c:
                print('- %d variables')
                for x in c:
                    print('- %s' % (x))
            if d:
                print('+ %d variables')
                for x in d:
                    print('+ %s' % (x))
        self._names = names


class KungfuLoadInitModelHook(tf.train.SessionRunHook):
    def __init__(self, **kwargs):
        print('%s created' % (self.__class__.__name__))
        self._loaded = False
        self._step = 0
        self._cycle = 0

    def begin(self):
        self._variables = tf.global_variables()
        print('model has %d variables' % (len(self._variables)))
        self._assign_ops = []
        self._placeholders = []
        for v in self._variables:
            p = tf.placeholder(v.dtype, v.shape)
            a = tf.assign(v, p)
            self._placeholders.append(p)
            self._assign_ops.append(a)

    def after_create_session(self, sess, coord):
        if self._loaded:
            return

        t0 = time.time()
        load_model(_MODEL_FILE, sess, self._variables, self._placeholders,
                   self._assign_ops)
        print('model loaded from %s, took %.2fs' %
              (_MODEL_FILE, time.time() - t0))
        self._loaded = True

    def before_run(self, run_context):
        pass
        # if self._step == 0:
        #     filename = checkpoint_name('before_run', self._cycle, self._step)
        #     save_model(filename, run_context.session, self._variables)

    def after_run(self, run_context, run_values):
        self._step += 1

    def end(self, sess):
        self._cycle += 1


def must_get_tensor_by_name(name):
    realname = name + ':0'
    options = []
    for v in tf.global_variables():
        if v.name == realname:
            options.append(v)
    [v] = options
    return v


class KungfuChangeBatchSizeHook(tf.train.SessionRunHook):
    def __init__(self, **kwargs):
        print('%s created' % (self.__class__.__name__))
        self._step = 0
        self._cycle = 0
        self._init_bs = 32
        self._bs = self._init_bs

        self._history_gns = []

    def begin(self):
        self._last_gns = must_get_tensor_by_name('monitor_grads_cond/last_gns')
        self._device_batch_size = must_get_tensor_by_name('device_batch_size')
        self._device_batch_size_place = tf.placeholder(tf.int32)
        self._set_device_batch_size = tf.assign(self._device_batch_size,
                                                self._device_batch_size_place)

    def after_create_session(self, sess, coord):
        pass

    def before_run(self, run_context):
        self._update_batch_size(run_context.session, self._bs)

        # show bs
        bs = run_context.session.run(self._device_batch_size)
        print('bs=%d' % (bs))

    def after_run(self, run_context, run_values):
        gns = self._get_last_gns(run_context.session)
        print('after step: %d, gns: %f' % (self._step, gns))
        self._step += 1

    def end(self, sess):
        record = tuple((self._cycle, self._step, self._get_last_gns(sess)))
        self._history_gns.append(record)
        pass
        # self._bs += 1
        self._cycle += 1

        if self._cycle == 19:
            for r in self._history_gns:
                print(r)

    def _get_last_gns(self, sess):
        return sess.run(self._last_gns)

    def _update_batch_size(self, sess, bs):
        feed_dict = {
            self._device_batch_size_place: bs,
        }
        sess.run(self._set_device_batch_size, feed_dict=feed_dict)
