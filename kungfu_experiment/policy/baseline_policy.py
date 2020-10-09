import kungfu.tensorflow as kf
import tensorflow as tf

from .base_policy import KungFuPolicy


class BaselinePolicy(KungFuPolicy):
    def __init__(self, epoch_size, epoch_num, init_batch_size, *args,
                 **kwargs):
        super(BaselinePolicy, self).__init__(epoch_size, epoch_num,
                                          init_batch_size, *args, **kwargs)
        self._bs = init_batch_size

    def before_train(self, vars, params):
        pass

    def _run_sync_op(self):
        pass

    def before_epoch(self, vars, params):
        pass

    def after_step(self, sess, vars, params, grads):
        pass

    def after_epoch(self, sess, vars, params):
        pass

    def get_batch_size(self):
        return self._bs
