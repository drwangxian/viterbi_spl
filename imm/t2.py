import numpy as np

p = 0.5

def p_norm_fn(x):

    x = np.abs(x)
    x = x ** p
    x = np.sum(x)
    x = x ** (1. / p)

    return x


aa = np.asarray([1., 2, 3.])
bb = np.asarray([3., 1.5, 2])
na = p_norm_fn(aa)
nb = p_norm_fn(bb)
nab = p_norm_fn(aa + bb)
assert na + nb >= nab, '{} vs {}'.format(na + nb, nab)

