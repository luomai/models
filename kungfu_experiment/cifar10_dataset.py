import tensorflow as tf
from official.resnet.cifar10_main import (_NUM_IMAGES, _RECORD_BYTES,
                                          get_filenames, parse_record)
from tensorflow.contrib.data.python.ops import threadpool


# rewite of official.resnet.resnet_run_loop.process_record_dataset
def process_record_dataset(dataset,
                           is_training,
                           offset,
                           batch_size,
                           shuffle_buffer,
                           parse_record_fn,
                           num_epochs=1,
                           dtype=tf.float32,
                           datasets_num_private_threads=None,
                           num_parallel_batches=1):

    # KungFu: Shard the dataset
    if is_training:
        from kungfu.python import current_cluster_size, current_rank
        dataset = dataset.shard(num_shards=current_cluster_size(),
                                index=current_rank())

    # Prefetches a batch at a time to smooth out the time taken to load input
    # files for shuffling and processing.
    dataset = dataset.prefetch(buffer_size=batch_size)

    if is_training:
        # Shuffles records before repeating to respect epoch boundaries.
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    # Repeats the dataset for the number of epochs to train.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.skip(offset)

    # Parses the raw records into images and labels.
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            lambda value: parse_record_fn(value, is_training, dtype),
            batch_size=batch_size,
            num_parallel_batches=num_parallel_batches,
            drop_remainder=False))

    # Operations between the final prefetch and the get_next call to the iterator
    # will happen synchronously during run time. We prefetch here again to
    # background all of the above processing work and keep it out of the
    # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
    # allows DistributionStrategies to adjust how many batches to fetch based
    # on how many devices are present.
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    # Defines a specific size thread pool for tf.data operations.
    if datasets_num_private_threads:
        tf.logging.info('datasets_num_private_threads: %s',
                        datasets_num_private_threads)
        dataset = threadpool.override_threadpool(
            dataset,
            threadpool.PrivateThreadPool(
                datasets_num_private_threads,
                display_name='input_pipeline_thread_pool'))

    return dataset


# rewite of official.resnet.cifar10_main.input_fn
def input_fn(is_training,
             data_dir,
             offset,
             batch_size,
             num_epochs=1,
             dtype=tf.float32,
             datasets_num_private_threads=None,
             num_parallel_batches=1):
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)

    return process_record_dataset(
        dataset=dataset,
        is_training=is_training,
        offset=offset,
        batch_size=batch_size,
        shuffle_buffer=_NUM_IMAGES['train'],
        parse_record_fn=parse_record,
        num_epochs=num_epochs,
        dtype=dtype,
        datasets_num_private_threads=datasets_num_private_threads,
        num_parallel_batches=num_parallel_batches)
