import os

import tensorflow as tf
from kungfu_experiment.cifar10_dataset import input_fn
from official.resnet import cifar10_main, resnet_model, resnet_run_loop


def train(data_dir, model_dir, init_batch_size, num_epochs, max_steps):
    batch_size = tf.placeholder(dtype=tf.int64)
    offset = tf.placeholder(dtype=tf.int64)
    input_ds = input_fn(True, data_dir, offset, batch_size, num_epochs)
    it = input_ds.make_initializable_iterator()
    features, labels = it.get_next()

    params = {
        'resnet_size': 56,
        'data_format': None,
        'batch_size': init_batch_size,
        'resnet_version': 1,
        'loss_scale': 1,
        'dtype': tf.float32,
        'fine_tune': False
    }
    print('creating train_step')
    spec = cifar10_main.cifar10_model_fn(features, labels,
                                         tf.estimator.ModeKeys.TRAIN, params)

    step_ops = [
        spec.train_op,
        spec.loss,
        spec.eval_metric_ops,
    ]

    def handle_step(step, _, loss, metrics):
        print('step: %d, loss: %f, metrics: %s' % (step, loss, metrics))

    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    current_offset = 0
    with tf.train.MonitoredSession() as sess:
        sess.run(init)
        sess.run(init_local)
        sess.run(it.initializer,
                 feed_dict={
                     batch_size: init_batch_size,
                     offset: current_offset,
                 })
        for step in range(max_steps):
            print('before step: %d' % (step))
            step_result = sess.run(step_ops)
            handle_step(step, *step_result)


def main():
    from kungfu import current_rank
    rank = current_rank()
    model_dir = os.path.join(os.getenv('HOME'), 'tmp/cifar10')
    model_dir = os.path.join(model_dir, str(rank))

    data_dir = os.path.join(os.getenv('HOME'), 'var/data/cifar')
    num_epochs = 2
    train_steps = 10000
    init_batch_size = 50
    train(data_dir, model_dir, init_batch_size, num_epochs, train_steps)


main()
