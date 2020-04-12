import time

import tensorflow as tf


class KungfuLogStepHook(tf.train.SessionRunHook):
    def __init__(self, **kwargs):
        self._step = 0

    def begin(self):
        self._t_begin = time.time()
        print('%s::%s(%d)' % (self.__class__.__name__, 'begin', self._step))

    def after_create_session(self, sess, coord):
        dur = time.time() - self._t_begin
        print('%s::%s(%d), took %.3f since begin' %
              (self.__class__.__name__, 'after_create_session', self._step, dur))

    def before_run(self, run_context):
        self._t_before = time.time()
        if self._step % 20 == 0:
            print('%s::%s(%d)' %
                  (self.__class__.__name__, 'before_run', self._step))

    def after_run(self, run_context, run_values):
        dur = time.time() - self._t_before
        if self._step % 20 == 0:
            print('%s::%s(%d), %.3fs per step' %
                  (self.__class__.__name__, 'after_run', self._step, dur))
        self._step += 1

    def end(self, sess):
        dur = time.time() - self._t_begin
        print('%s::%s(%d), took %.2fs' %
              (self.__class__.__name__, 'end', self._step, dur))
