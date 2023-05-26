import numpy as np

def gen_transition_matrix_fn(bins_per_semitone, n_bins):

    probs_of_distance = np.arange(n_bins) // bins_per_semitone
    probs_of_distance = np.exp(-probs_of_distance)
    cutoff = 10 * bins_per_semitone
    probs_of_distance[cutoff:] = probs_of_distance[cutoff - 1]

    range_n_bins = np.arange(n_bins)
    distance_matrix = range_n_bins[:, None] - range_n_bins[None, :]
    distance_matrix = np.abs(distance_matrix)
    transition_matrix = np.empty([n_bins + 1, n_bins + 1], np.float64)
    transition_matrix[:n_bins, :n_bins] = probs_of_distance[distance_matrix]

    cp = probs_of_distance[cutoff - 1]
    pf_0 = cp * 10 ** (-90)
    p0_0 = cp * 10 ** (-100)
    p0_f = cp * 10 ** (-80)
    transition_matrix[:n_bins, n_bins] = pf_0
    transition_matrix[n_bins, :n_bins] = p0_f
    transition_matrix[n_bins, n_bins] = p0_0
    transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1)[:, None]
    t = np.sum(transition_matrix, axis=1)
    assert np.allclose(t, 1.)

    return transition_matrix

def durrieu_gen_transition_matrix_fn(bins_per_semitone, n_bins):

    stepNotes = bins_per_semitone
    NF0 = n_bins

    # Viterbi decoding to estimate the predominant fundamental
    # frequency line
    scale = 1.0
    # transitions = np.exp(-np.floor(np.arange(0,NF0) / stepNotes) * scale)
    transitions = np.arange(NF0) // stepNotes
    transitions = np.exp(-transitions * scale)
    cutoffnote = 2 * 5 * stepNotes
    transitions[cutoffnote:] = transitions[cutoffnote - 1]

    transitionMatrixF0 = np.zeros([NF0 + 1, NF0 + 1]) # toeplitz matrix
    b = np.arange(NF0)
    # transitionMatrixF0[0:NF0, 0:NF0] = \
    #                           transitions[\
    #     np.array(np.abs(np.outer(np.ones(NF0), b) \
    #                     - np.outer(b, np.ones(NF0))), dtype=int)]

    _distance_matrix = b[:, None] - b[None, :]
    _distance_matrix = np.abs(_distance_matrix)
    assert np.issubdtype(_distance_matrix.dtype, np.signedinteger)
    transitionMatrixF0[:NF0, :NF0] = transitions[_distance_matrix]

    pf_0 = transitions[cutoffnote - 1] * 10 ** (-90)
    p0_0 = transitions[cutoffnote - 1] * 10 ** (-100)
    p0_f = transitions[cutoffnote - 1] * 10 ** (-80)
    transitionMatrixF0[:NF0, NF0] = pf_0
    transitionMatrixF0[NF0, :NF0] = p0_f
    transitionMatrixF0[NF0, NF0] = p0_0

    sumTransitionMatrixF0 = np.sum(transitionMatrixF0, axis=1)
    transitionMatrixF0 = transitionMatrixF0 / sumTransitionMatrixF0[:, None]

    return transitionMatrixF0


if __name__ == '__main__':

    b = 20
    n_bins = 721

    m = gen_transition_matrix_fn(b, n_bins)
    _m = durrieu_gen_transition_matrix_fn(b, n_bins)
    t = np.sum(np.abs(m - _m))
    print(t)







