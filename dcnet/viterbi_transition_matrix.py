from self_defined import load_np_array_from_file_fn, save_np_array_to_file_fn
import numpy as np
import librosa

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

file_name = 'transition_int.dat'
name, transition = load_np_array_from_file_fn(file_name)
assert name == file_name[:-4]

# file_name = 'p_steady.dat'
# name, p_steady = load_np_array_from_file_fn(file_name)
# assert name == file_name[:-4]
#
#
# p_th = 3e-3  # account for 70% of voiced states
#
# d_max = 12
# d_trans_list = []
# p_trans_list = []
# p_voiced = 1. - p_steady[-1]
#
# n_bins = 320
# for i in range(n_bins):
#     if p_steady[i] < p_th:
#         continue
#     p_trans_list.append(p_steady[i])
#     d_trans = np.zeros([2 * d_max + 1], np.int64)
#     for j in range(n_bins):
#         if transition[i, j]:
#             d = j - i
#             d = max(d, -d_max)
#             d = min(d, d_max)
#             d += d_max
#             d_trans[d] += transition[i, j]
#     # d_trans = d_trans / np.sum(d_trans)
#     d_trans_list.append(d_trans)
#     print('bin - {}'.format(i))
#     print_fn(d_trans)

# d_trans = np.stack(d_trans_list, axis=0)
# ps = np.asarray(p_trans_list)
# ps = ps / np.sum(ps)
# d_trans = d_trans * ps[:, None]
# d_trans = np.sum(d_trans, axis=0)
# print('final')
# print_fn(d_trans)

d_max = 12
d_trans = np.zeros([2 * d_max + 1], np.int64)

n_bins = 320
for i in range(n_bins):
    for j in range(n_bins):
        if transition[i, j]:
            d = j - i
            d = max(d, -d_max)
            d = min(d, d_max)
            d = d + d_max
            d_trans[d] += transition[i, j]
d_trans = np.maximum(d_trans, 6)
d_trans = d_trans / np.sum(d_trans)

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









