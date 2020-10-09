import kungfu.tensorflow as kf
import tensorflow as tf
from kungfu.tensorflow.policy import BasePolicy


class BaselinePolicy(BasePolicy):
    def __init__(self, init_batch_size):
        self._bs = init_batch_size

    def before_train(self):
        pass

    def before_epoch(self, sess):
        pass

    def after_step(self, sess):
        pass

    def after_epoch(self, sess):
        pass

    def get_batch_size(self):
        return self._bs
