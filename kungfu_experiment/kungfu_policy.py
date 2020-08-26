class KungFuPolicy(object):
    def before(self):
        pass

    def after(self, gradients, metrics):
        pass


import tensorflow as tf


class KungfuAdaptationHook(tf.train.SessionRunHook):
    def __init__(self):
        policy = KungFuPolicy()
        self._policy = policy

    def before_run(self, run_context):
        self._policy.before()

    def after_run(self, run_context, run_values):
        gradients = None
        metrics = None
        self._policy.after(gradients, metrics)


def must_get_tensor_by_name(name):
    realname = name + ':0'
    options = []
    for v in tf.global_variables():
        if v.name == realname:
            options.append(v)
    [v] = options
    return v


class EMA:
    def __init__(self, alpha, scale_cap):
        self._alpha = alpha
        self._value = None
        self._scale_cap = scale_cap

    def _cap(self, x):
        up = self._value * self._scale_cap
        if x > up:
            return up
        down = self._value / self._scale_cap
        if x < down:
            return down
        return x

    def update(self, x):
        if self._value is None:
            self._value = x
        else:
            x = self._cap(x)
            self._value = self._alpha * self._value + (1 - self._alpha) * x
        return self._value

    def get(self):
        return self._value

    def reset(self):
        self._value = None


class KungfuChangeBatchSizeHook(tf.train.SessionRunHook):
    def __init__(self, **kwargs):
        print('%s created' % (self.__class__.__name__))
        self._step = 0
        self._cycle = 0
        self._init_bs = 32
        self._max_bs = 1024
        self._bs = self._init_bs

        self._history_gns = []
        self._ratio_ema = EMA(0.9, 2)
        self._history_ratio_ema = []

    def begin(self):
        self._last_gns = must_get_tensor_by_name('monitor_grads_cond/last_gns')
        self._device_batch_size = must_get_tensor_by_name('device_batch_size')
        self._device_batch_size_place = tf.placeholder(tf.int32)
        self._set_device_batch_size = tf.assign(self._device_batch_size,
                                                self._device_batch_size_place)

    def after_create_session(self, sess, coord):
        pass

    def before_run(self, run_context):
        self._update_batch_size(run_context.session, self._bs)

        # show bs
        bs = run_context.session.run(self._device_batch_size)
        print('bs=%d' % (bs))

    def after_run(self, run_context, run_values):
        bs = run_context.session.run(self._device_batch_size)
        gns = self._get_last_gns(run_context.session)
        gns_abs = abs(gns)
        ratio = gns_abs / bs
        self._ratio_ema.update(ratio)
        print('after step: %d, |gns|: %f, bs=%d, |gns|/bs: %f, ratio_ema: %f' %
              (self._step, gns_abs, bs, ratio, self._ratio_ema.get()))
        self._step += 1

    def _normalize_bs_scale(self, bs_scale):
        if bs_scale > 2:
            return 2
        if bs_scale < 1:
            return 1
        return bs_scale

    def end(self, sess):
        record = tuple((self._cycle, self._step, self._get_last_gns(sess)))
        self._history_gns.append(record)
        self._history_ratio_ema.append(self._ratio_ema.get())

        # self._bs += 1
        self._cycle += 1

        print('history_ratio_ema: %s' % (self._history_ratio_ema))
        if len(self._history_ratio_ema) > 1:
            bs_scale = self._history_ratio_ema[-1] / self._history_ratio_ema[0]
            normalized_bs_scale = self._normalize_bs_scale(bs_scale)
            print('normalized bs scale: %f -> %f' %
                  (bs_scale, normalized_bs_scale))

            new_bs = min(self._max_bs, int(self._bs * normalized_bs_scale))
            print('will change bs %d -> %d' % (self._bs, new_bs))
            self._bs = new_bs

        if self._cycle == 19:
            for r in self._history_gns:
                print(r)

    def _get_last_gns(self, sess):
        return sess.run(self._last_gns)

    def _update_batch_size(self, sess, bs):
        feed_dict = {
            self._device_batch_size_place: bs,
        }
        sess.run(self._set_device_batch_size, feed_dict=feed_dict)
