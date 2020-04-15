import time

import numpy as np
import tensorflow as tf

USE_DYNAMIC_BATCH_SIZE = False
# USE_DYNAMIC_BATCH_SIZE = True


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


class KungfuLoadInitModelHook(tf.train.SessionRunHook):
    def __init__(self, **kwargs):
        print('%s created' % (self.__class__.__name__))
        pass

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
        t0 = time.time()
        load_model(_MODEL_FILE, sess, self._variables, self._placeholders,
                   self._assign_ops)
        print('model loaded from %s, took %.2fs' %
              (_MODEL_FILE, time.time() - t0))
