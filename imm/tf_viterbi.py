import numpy as np
import tensorflow as tf





def tf_viterbi_librosa_fn(*, tf_log_transition_matrix_T, tf_log_prob_init, tf_or_np_log_probs_st):

    """
    This has been optimized by shaun.
    Args:
        B: S * S, sum(B, axis=1) = 1
        prob_init: S, sum(S) = 1
        probs: S * T, prob(observation at t| in state s at t)

    Returns:

    """

    B = tf_log_transition_matrix_T  # Target <-- Source
    probs = tf.convert_to_tensor(tf_or_np_log_probs_st, tf.float32)  # R ^ {S * T}
    prob_init = tf_log_prob_init

    S = len(B)
    assert B.shape == (S, S)
    assert len(prob_init) == S
    T = probs.shape[1]
    assert probs.shape == (S, T)

    probs = tf.transpose(probs)

    T1 = np.empty([T, S], np.float32)
    T2 = np.empty([T, S], np.int32)

    Bt_values = prob_init + probs[0]
    T1[0] = Bt_values

    @tf.function(
        input_signature=[
            tf.TensorSpec([S], name='')
        ]
    )
    def _tf_core_fn(Bt_values, probs):

        Bt = B + Bt_values
        Bt_indices = tf.argmax(Bt, axis=1, output_type=tf.int32)
        Bt_values = tf.gather(Bt, axis=1, batch_dims=1, indices=Bt_indices) + probs

        return Bt_indices, Bt_values







    for t in range(1, T):
        Bt = B + Bt_values
        Bt_indices = tf.argmax(Bt, axis=1, output_type=tf.int32)
        T2[t] = Bt_indices
        Bt_values = tf.gather(Bt, axis=1, batch_dims=1, indices=Bt_indices) + probs[t]
        T1[t] = Bt_values

    states = np.empty([T], np.int32)
    s = tf.argmax(Bt_values)
    states[-1] = s
    for t in range(T - 2, -1, -1):
        s = T2[t + 1, s]
        states[t] = s

    return states


def viterbi_librosa_fn(*, log_transition_matrix_T, log_prob_init, log_probs_st):

    B = log_transition_matrix_T
    assert B.flags['C_CONTIGUOUS'] == True
    assert B.dtype == np.float32
    S = len(B)

    assert len(log_prob_init) == S
    assert log_probs_st.dtype == np.float32
    prob_init = log_prob_init

    assert log_probs_st.shape[0] == S
    assert log_probs_st.dtype == np.float32
    T = log_probs_st.shape[1]
    probs = np.require(log_probs_st.T, requirements=['C'])

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