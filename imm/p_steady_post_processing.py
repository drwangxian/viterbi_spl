import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from self_defined import load_np_array_from_file_fn, save_np_array_to_file_fn
import numpy as np

name, p_steady = load_np_array_from_file_fn('p_steady.dat')
assert name == 'p_steady'
assert np.all(p_steady >= 0)
assert np.isclose(np.sum(p_steady), 1)
p_th = 1. / len(p_steady) / 10.

p_unvoiced = p_steady[-1]
p_voiced = 1. - p_unvoiced

ps = p_steady[:-1]
ps = np.maximum(ps, p_th)
ps = ps / np.sum(ps)
ps = ps * p_voiced
_p_steady = np.append(ps, p_unvoiced)
_p_steady = _p_steady.astype(np.float32)
t = np.sum(_p_steady)
assert np.isclose(t, 1.)
save_np_array_to_file_fn('viterbi_init_probs.dat', _p_steady, 'viterbi_init_probs')

p_steady_log = np.log10(_p_steady)

plt.plot(p_steady_log)
plt.savefig('p_steady_log.png')
plt.close()