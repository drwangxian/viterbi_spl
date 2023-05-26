from self_defined import load_np_array_from_file_fn, save_np_array_to_file_fn
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

N_BINS = 721
BINS_PER_OCT = 240


def single_side_d_max_fn(h, B):

    # h in second
    # B bins_per_octave
    d_max = 35.92 * h * B * 1.3 // 2
    d_max = int(d_max)

    return d_max


def print_fn(d_trans):

    d = len(d_trans)
    d = d // 2
    d_trans = np.round(d_trans, 5)
    d_trans = d_trans * 1e5
    d_trans = d_trans.astype(np.int32)
    for i in range(2 * d + 1):
        print('{:-5d}  '.format(i - d), end='')
    print()
    for v in d_trans:
        print('{:-5d}  '.format(v), end='')
    print()
    print()

n_bins = N_BINS

file_name = 'transition_int.dat'
name, transition = load_np_array_from_file_fn(file_name)
assert name == file_name[:-4]
assert transition.shape == (n_bins + 1, n_bins + 1)


d_max = single_side_d_max_fn(h=0.01, B=BINS_PER_OCT)
d_trans = np.zeros([2 * d_max + 1], np.int64)

for i in range(n_bins):
    for j in range(n_bins):
        if transition[i, j]:
            d = j - i
            d = max(d, -d_max)
            d = min(d, d_max)
            d = d + d_max
            d_trans[d] += transition[i, j]
d_trans = np.maximum(d_trans, 2)
d_trans = d_trans / np.sum(d_trans)

t = np.log10(d_trans)
t1 = np.arange(-d_max, d_max + 1)
plt.scatter(t1, t, s=3, color='k')
plt.savefig('d_trans.png')
plt.close()

transition_matrix = np.zeros([n_bins, n_bins], np.float32)
switch = np.asarray([[0.98713454, 0.01286546],
                     [0.01002112, 0.98997888]], np.float32)

for i in range(n_bins):
    for j in range(n_bins):
        d = np.abs(j - i)
        if d > d_max:
            continue

        d = j - i
        d += d_max
        transition_matrix[i, j] = d_trans[d]

transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1)[:, None]
t = np.sum(transition_matrix, axis=1)
assert np.all(np.isclose(t, 1))
transition_matrix = np.pad(transition_matrix, [(0, 1), (0, 1)])
transition_matrix[:n_bins, :n_bins] *= switch[0, 0]
transition_matrix[:n_bins, n_bins] = switch[0, 1]
transition_matrix[n_bins, :n_bins] = switch[1, 0] / n_bins
transition_matrix[n_bins, n_bins] = switch[1, 1]
t = np.sum(transition_matrix, axis=1)
assert np.all(np.isclose(t, 1))
save_np_array_to_file_fn('viterbi_transition_matrix.dat', transition_matrix, 'viterbi_transition_matrix')









