import os

import tensorflow as tf

from official.resnet import cifar10_main, resnet_model, resnet_run_loop


def main():
    init_batch_size = 50
    data_dir = os.path.join(os.getenv('HOME'), 'var/data/cifar')

    batch_size = tf.Variable(init_batch_size, dtype=tf.int64, trainable=False)
    input_ds = cifar10_main.input_fn(True, data_dir, batch_size)
    print(input_ds)
    it = input_ds.make_initializable_iterator()
    features, labels = it.get_next()

    mode = tf.estimator.ModeKeys.TRAIN
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
    spec = cifar10_main.cifar10_model_fn(features, labels, mode, params)
    train_op, eval_metric_ops = spec.train_op, spec.eval_metric_ops


    # print('train_step is')
    # print(train_op)
    # print('eval_metric_ops is')
    # print(eval_metric_ops)
    max_steps = 10

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        sess.run(it.initializer)
        for step in range(max_steps):
            print('before step: %d' % (step))
            sess.run(train_op)
            # print('before eval: %d' % (step))
            # metrics = sess.run(eval_metric_ops)
            # print(metrics)


main()
