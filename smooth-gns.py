#!/usr/bin/env python3


def list_gns():
    for line in open('gns.txt'):
        yield abs(float(line.strip()))


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


def smooth(xs, alpha, scale_cap):
    ys = []
    ema = EMA(alpha, scale_cap)
    for x in xs:
        ys.append(ema.update(x))
    return ys


gns = list(list_gns())

alpha = 0.995
scale_cap = 3
ys = smooth(gns, alpha, scale_cap)

for y in ys:
    print(y)

# print(gns)
