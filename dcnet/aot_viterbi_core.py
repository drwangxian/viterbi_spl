import numpy as np
from numba.pycc import CC

cc = CC('viterbi_numba')
cc.verbose = True


@cc.export('core', 'i8[:](f4[:, ::1], f4[:], f4[:, ::1])')
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

    B[:] = np.log(B + tinyp)
    prob_init[:] = np.log(prob_init + tinyp)
    probs[:] = np.log(probs + tinyp)

    T1 = np.empty((T, S), np.float32)
    T2 = np.empty((T, S), np.int64)

    T1[0] = prob_init + probs[0]

    Bt = np.empty((S, S), np.float32)

    for t in range(1, T):
        _B = T1[t - 1] + B
        all_s = np.argmax(_B, axis=1)
        T2[t] = all_s
        _B = np.take_along_axis(_B, np.expand_dims(all_s, axis=-1), axis=1)
        T1[t] = _B[:, 0] + probs[t]

        np.add(T1[t - 1], B, out=Bt)
        np.argmax(Bt, axis=1, out=T2[t])
        np.add(np.take_along_axis(Bt, T2[t][:, None], axis=1)[:, 0], probs[t], out=T1[t])



    states = np.empty((T,), np.int64)
    s = np.argmax(T1[-1])
    states[-1] = s
    for t in range(T - 2, -1, -1):
        s = T2[t + 1, s]
        states[t] = s

    return states


if __name__ == "__main__":
    cc.compile()
