import kungfu.tensorflow as kf
import tensorflow as tf
from kungfu.python import current_cluster_size
from kungfu.tensorflow.ops import all_reduce
from kungfu.tensorflow.policy import BasePolicy
from kungfu.utils import EMA


class GNSPolicy(BasePolicy):
    def __init__(self, init_batch_size):
        self._ratio_ema = EMA(0.9, 2)
        self._prev_ave = None
        self._prev = None
        self._bs = init_batch_size

    def before_train(self):
        self._gns_place = tf.placeholder(dtype=tf.float32)
        self._gns_ave = all_reduce(self._gns_place)
        self._need_sync = True  # Worker state synchronisation flag
        self._set_batch_size = kf.create_setter(kf.get_or_create_batch_size())

    def _run_sync_op(self):
        pass

    def before_epoch(self, sess):
        self._bs = kf.eval_batch_size(sess)
        if self._need_sync:
            self._run_sync_op()
            self._need_sync = False

    def after_step(self, sess):
        bs = kf.eval_batch_size(sess)
        gns = kf.eval_gradient_noise_scale(sess)  #  get monitored valued
        gns_abs = abs(gns)
        self._ratio_ema.update(gns_abs / bs)
        print('|gns| = %s, ems: %s' % (gns_abs, self._ratio_ema.get()))

    def after_epoch(self, sess):
        # print('%s' % ('after_epoch'))
        gns_ema = self._ratio_ema.get()

        # TODO: run all_reduce without operator
        ave = sess.run(self._gns_ave, feed_dict={self._gns_place: gns_ema})
        print('after epoch: ave: %s' % (ave))
        if self._prev_ave is not None:
            bs = kf.eval_batch_size(sess)
            if ave > self._prev_ave:
                new_bs = min(bs * 2, 1024)
                print('change batch size %d -> %d' % (bs, new_bs))
                self._set_batch_size(sess, new_bs)
                self._bs = new_bs
        self._prev_ave = ave

    def get_batch_size(self):
        return self._bs
