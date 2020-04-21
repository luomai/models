from kungfu._utils import map_maybe
from kungfu.tensorflow.ops import (counter, current_cluster_size, current_rank,
                                   fuse, global_noise_scale, group_all_reduce)
from kungfu.tensorflow.optimizers.core import (_create_kungfu_optimizer,
                                               _KungFuAlgorithm)

import tensorflow as tf


def MonitorGradientNoiseScaleOptimizer(optimizer,
                                       device_batch_size,
                                       monitor_interval=1,
                                       name=None,
                                       use_locking=False):
    mon_gns_algo = _GradientNoiseScale(device_batch_size, monitor_interval)
    return _create_kungfu_optimizer(optimizer, mon_gns_algo, name, use_locking)


class _GradientNoiseScale(_KungFuAlgorithm):
    def __init__(self, device_batch_size, monitor_interval=1):
        self._num_workers = current_cluster_size()
        self._step = counter()

        self._interval = monitor_interval
        self._device_batch_size = tf.cast(device_batch_size, dtype=tf.float32)
        self._global_batch_size = self._device_batch_size * self._num_workers

    def _monitor(self, grads, reduced_grads):
        # Only the master node is doing the global monitoring.
        noise_op = global_noise_scale(self._device_batch_size,
                                      self._global_batch_size, fuse(grads),
                                      fuse(reduced_grads))

        print_op = tf.print('Gradient Noise Scale:', noise_op)
        return print_op

    def apply_gradients(self, apply_grads_func, grads_and_vars, **kwargs):
        grads, variables = list(zip(*grads_and_vars))

        # Synchronization logic
        summed_grads = group_all_reduce(grads)
        reduced_grads = map_maybe(lambda g: g / self._num_workers,
                                  summed_grads)

        # Monitoring logic
        monitor_grads_op = tf.cond(
            tf.equal(tf.mod(self._step, self._interval), 0),
            lambda: self._monitor(grads, reduced_grads), lambda: tf.no_op())

        with tf.control_dependencies([monitor_grads_op]):
            return apply_grads_func(zip(reduced_grads, variables), **kwargs)
