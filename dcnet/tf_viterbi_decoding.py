import tensorflow as tf
import numpy as np
from self_defined import load_np_array_from_file_fn
import numba
import time
import viterbi_numba

name, transition_matrix = load_np_array_from_file_fn('tf_transition_matrix.dat')
assert name == 'tf_transition_matrix'
name, prob_init = load_np_array_from_file_fn('tf_prob_init.dat')
assert name == 'tf_prob_init'
name, probs_st = load_np_array_from_file_fn('tf_probs_st.dat')
assert name == 'tf_probs_st'
assert probs_st.flags['F_CONTIGUOUS']

S = 321
T = probs_st.shape[1]

T1 = tf.Variable(tf.zeros([T, S]), trainable=False)
T2 = tf.Variable(tf.zeros([T, S], tf.int32), trainable=False)
states = tf.Variable(tf.zeros([T], tf.int32))

@tf.function(
    input_signature=[
        tf.TensorSpec([321, 321], name='transition_matrix'),
        tf.TensorSpec([321], name='prob_init'),
        tf.TensorSpec([321, None], name='probs_st')
    ],
    autograph=True
)
def viterbi_tf_fn(transition_matrix, prob_init, probs_st):

    S = 321
    transition_matrix = tf.convert_to_tensor(transition_matrix, tf.float32)
    transition_matrix.set_shape([S, S])

    prob_init = tf.convert_to_tensor(prob_init, tf.float32)
    prob_init.set_shape([S])

    probs_st = tf.convert_to_tensor(probs_st, tf.float32)
    probs_st.set_shape([S, None])
    T = tf.shape(probs_st)[1]

    B = transition_matrix
    t = tf.reduce_sum(B, axis=1)
    tf.debugging.assert_near(t, 1.)
    t = tf.reduce_sum(prob_init)
    tf.debugging.assert_near(t, 1.)

    tinyp = np.finfo(np.float32).tiny
    tinyp = tf.convert_to_tensor(tinyp)

    B = tf.math.log(tf.transpose(B) + tinyp)  # S <- S
    prob_init = tf.math.log(prob_init + tinyp)
    probs = tf.math.log(tf.transpose(probs_st) + tinyp)  # T * S

    T1.scatter_update(tf.IndexedSlices(values=prob_init + probs[0], indices=0))

    for t in tf.range(1, T, dtype=tf.int32):
        _B = T1[t - 1] + B
        all_s = tf.argmax(_B, axis=1, output_type=tf.int32)
        T2.scatter_update(tf.IndexedSlices(values=all_s, indices=t))
        _B = tf.gather_nd(_B, indices=all_s[:, None], batch_dims=1)
        T1.scatter_update(tf.IndexedSlices(values=_B + probs[t], indices=t))

    s = tf.argmax(T1[-1], output_type=tf.int32)
    states.scatter_update(tf.IndexedSlices(indices=T - 1, values=s))
    for t in tf.range(T - 2, -1, -1, dtype=tf.int32):
        s = T2[t + 1, s]
        states.scatter_update(tf.IndexedSlices(indices=t, values=s))

    return states


@numba.jit(
    signature_or_function=numba.int64[:](numba.float32[:, ::1], numba.float32[:], numba.float32[:, ::1]),
    cache=True,
    nopython=True
)
def _viterbi_core_numba_fn(B, prob_init, probs):

    """
    B: S <- S, c-flag
    prob_init: S
    probs: T, S, c-flag
    """

    tinyp = 1.1754944e-38

    S = B.shape[0]
    T = probs.shape[0]
    assert prob_init.shape[0] == S

    B = np.log(B + tinyp)
    prob_init = np.log(prob_init + tinyp)
    probs = np.log(probs + tinyp)

    T1 = np.empty((T, S), np.float32)
    T2 = np.empty((T, S), np.int64)

    T1[0] = prob_init + probs[0]

    for t in range(1, T):
        _B = T1[t - 1] + B
        all_s = np.argmax(_B, axis=1)
        T2[t] = all_s
        _B = np.take_along_axis(_B, np.expand_dims(all_s, axis=-1), axis=1)
        T1[t] = _B[:, 0] + probs[t]
    states = np.empty((T,), np.int64)
    s = np.argmax(T1[-1])
    states[-1] = s
    for t in range(T - 2, -1, -1):
        s = T2[t + 1, s]
        states[t] = s

    return states


def viterbi_numba_fn(*, transition_matrix, prob_init, probs_st):

    """

    Args:
        B: S * S, sum(B, axis=1) = 1
        prob_init: S, sum(S) = 1
        probs: S * T, prob(observation at t| in state s at t)

    Returns:

    """

    B = transition_matrix
    probs = probs_st

    S = len(B)
    T = probs.shape[1]
    assert B.shape == (S, S)
    assert probs.shape == (S, T)
    t = np.sum(B, axis=1)
    assert np.allclose(t, 1.)
    assert len(prob_init) == S
    assert np.isclose(np.sum(prob_init), 1.)

    B = np.require(B.T, requirements=['C'])
    probs = np.require(probs.T, requirements=['C'])

    states = viterbi_numba.core(
        B,
        prob_init.copy(),
        probs
    )

    return states


def viterbi_librosa_c_fn(*, transition_matrix, prob_init, probs_st):

    """

    Args:
        B: S * S, sum(B, axis=1) = 1
        prob_init: S, sum(S) = 1
        probs: S * T, prob(observation at t| in state s at t)

    Returns:

    """

    B = transition_matrix
    probs = probs_st

    S = len(B)
    T = probs.shape[1]
    assert B.shape == (S, S)
    assert probs.shape == (S, T)
    t = np.sum(B, axis=1)
    assert np.allclose(t, 1.)
    assert len(prob_init) == S
    assert np.isclose(np.sum(prob_init), 1.)

    tinyp = np.finfo(probs.dtype).tiny

    B = np.log(B.T + tinyp)  # S * S
    B = np.require(B, requirements=['C'])
    prob_init = np.log(prob_init + tinyp)
    probs = np.log(probs.T + tinyp)  # T * S
    probs = np.require(probs, requirements=['C'])

    T1 = np.empty([T, S], np.float32)
    T2 = np.empty([T, S], np.int64)

    T1[0] = prob_init + probs[0]

    Bt = np.empty([S, S], np.float32)
    for t in range(1, T):
        np.add(T1[t - 1], B, out=Bt)
        np.argmax(Bt, axis=1, out=T2[t])
        np.add(np.take_along_axis(Bt, indices=T2[t][:, None], axis=1)[:, 0], probs[t], out=T1[t])

    states = np.empty([T], np.int64)
    s = np.argmax(T1[-1])
    states[-1] = s
    for t in range(T - 2, -1, -1):
        s = T2[t + 1, s]
        states[t] = s

    return states

def viterbi_librosa_fn(*, transition_matrix, prob_init, probs_st):

    """

    Args:
        B: S * S, sum(B, axis=1) = 1
        prob_init: S, sum(S) = 1
        probs: S * T, prob(observation at t| in state s at t)

    Returns:

    """

    B = transition_matrix
    probs = probs_st

    S = len(B)
    T = probs.shape[1]
    assert B.shape == (S, S)
    assert probs.shape == (S, T)
    t = np.sum(B, axis=1)
    assert np.allclose(t, 1.)
    assert len(prob_init) == S
    assert np.isclose(np.sum(prob_init), 1.)

    tinyp = np.finfo(probs.dtype).tiny

    B = np.log(B.T + tinyp)  # S * S
    B = np.require(B, requirements=['C'])
    prob_init = np.log(prob_init + tinyp)
    probs = np.log(probs.T + tinyp)  # T * S
    probs = np.require(probs, requirements=['C'])

    T1 = np.empty([T, S])
    T2 = np.empty([T, S], np.int32)

    T1[0] = prob_init + probs[0]

    for t in range(1, T):
        _B = T1[t - 1] + B
        all_s = np.argmax(_B, axis=1)
        T2[t] = all_s
        T1[t] = _B[np.arange(S), all_s] + probs[t]

    states = []
    s = np.argmax(T1[-1])
    p = T1[-1, s]
    states.append(s)
    for t in range(T - 2, -1, -1):
        s = T2[t + 1, s]
        states.append(s)
    states = states[::-1]
    states = np.asarray(states)

    return states


t0 = time.time()
states_c = viterbi_librosa_c_fn(
    transition_matrix=transition_matrix,
    prob_init=prob_init,
    probs_st=probs_st
)
t0 = time.time() - t0
print('c - {}'.format(t0))

t0 = time.time()
states_python = viterbi_librosa_fn(
    transition_matrix=transition_matrix,
    prob_init=prob_init,
    probs_st=probs_st
)
t0 = time.time() - t0
print('python - {}'.format(t0))

assert np.all(states_c == states_python)





