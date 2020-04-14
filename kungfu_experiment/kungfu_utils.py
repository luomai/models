import time

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
        print('%s::%s(%d)' % (self.__class__.__name__, 'begin', self._global_step))

    def after_create_session(self, sess, coord):
        dur = time.time() - self._t_begin
        print('%s::%s(%d), took %.3f since begin' %
              (self.__class__.__name__, 'after_create_session', self._global_step, dur))

    def before_run(self, run_context):
        self._t_before = time.time()
        if self._iter_step % 20 == 0:
            print('%s::%s(%d, %d)' %
                  (self.__class__.__name__, 'before_run', self._iter_step, self._global_step))

    def after_run(self, run_context, run_values):
        dur = time.time() - self._t_before
        if self._iter_step % 20 == 0:
            print('%s::%s(%d, %d), %.3fs per step' %
                  (self.__class__.__name__, 'after_run', self._iter_step, self._global_step, dur))
        self._global_step += 1
        self._iter_step += 1

    def end(self, sess):
        dur = time.time() - self._t_begin
        print('%s::%s(%d), took %.2fs' %
              (self.__class__.__name__, 'end', self._global_step, dur))
