import os.path

import logging
import numpy as np
import soundfile
import tensorflow as tf
from wf0 import GFMSpec
from wgamma import gen_WGAMMA_fn
import librosa
from tf_stft_istft import TF_STFT_ISTFT
from transition_matrix import gen_transition_matrix_fn
from tf_viterbi import tf_viterbi_librosa_fn
import soundfile
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import librosa.display


class Config:

    def __init__(self):

        self.w = 2048
        self.h = 256
        self.fs = 44100

        self.niters = 100
        self.patient_iters = 2

        self.R = 40
        self.P = 30
        self.K = 10
        self.F = self.w // 2 + 1
        self.fmin = 100
        self.fmax = 800
        self.bins_per_note = 20

        self.Oq = .25
        self.gfm_constant_E = True

        U = 12 * self.bins_per_note * np.log2(float(self.fmax) / self.fmin)
        U = np.ceil(U)
        U = int(U) + 1
        self.U = U


class Viterbi:


    def __init__(self, bins_per_semitone, n_bins):

        self.b = bins_per_semitone
        self.n_bins = n_bins

        transition_matrix = gen_transition_matrix_fn(bins_per_semitone=self.b, n_bins=self.n_bins)
        assert np.all(transition_matrix > 0)
        transition_matrix = np.log(transition_matrix.T)
        assert not np.any(np.isneginf(transition_matrix))
        transition_matrix = np.require(transition_matrix, np.float32, ['C'])
        self.log_transition_matrix_T = transition_matrix

        init_probs = np.empty([n_bins + 1])
        t = 1. / (n_bins + 1)
        init_probs.fill(t)
        init_probs = np.log(init_probs)
        init_probs = init_probs.astype(np.float32)
        self.log_prob_init = init_probs

    def process_HF0_fn(self, HF0):

        assert isinstance(HF0, (np.ndarray, tf.Tensor))
        if isinstance(HF0, tf.Tensor):
            HF0 = HF0.numpy()

        U = self.n_bins
        assert HF0.shape[0] == U

        t = HF0[HF0 > 0]
        t = t.min()
        if np.log(t) < -87:
            t = np.exp(-87)
        HF0 = HF0 + t
        HF0 = np.log(HF0)
        _min = np.min(HF0)
        HF0 = np.pad(HF0, [[0, 1], [0, 0]], mode='constant', constant_values=_min)

        return HF0

    def viterbi_librosa_fn(self, log_HF0):

        """
        np version is faster than tf version

        """

        S = self.n_bins + 1
        B = self.log_transition_matrix_T
        prob_init = self.log_prob_init

        assert isinstance(log_HF0, np.ndarray)
        assert log_HF0.dtype == np.float32
        assert log_HF0.shape[0] == S
        log_HF0 = np.transpose(log_HF0)
        log_HF0 = np.require(log_HF0, np.float32, ['C'])
        probs = log_HF0
        T = len(probs)

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

    def __call__(self, HF0):

        log_HF0 = self.process_HF0_fn(HF0)

        states = self.viterbi_librosa_fn(log_HF0)

        return states


class IMM(Config):

    def __init__(self):

        super(IMM, self).__init__()

        U = self.U
        fmin = self.fmin
        bins_per_oct = 12 * self.bins_per_note

        self.f0s = float(fmin) * 2. ** (np.arange(U) / float(bins_per_oct))
        self.WF0 = self.gen_WF0_fn()
        self.WGAMMA = gen_WGAMMA_fn(
            n_freq_bins=self.F,
            n_bases=self.P,
            overlap=0.75
        )
        self.eps = 1e-20

        self.tf_WF0 = tf.convert_to_tensor(self.WF0, tf.float32)
        self.tf_WF0T = tf.transpose(self.tf_WF0)

        self.tf_WGAMMA = tf.convert_to_tensor(self.WGAMMA, tf.float32)
        self.tf_WGAMMAT = tf.transpose(self.tf_WGAMMA)
        self.tf_eps = tf.convert_to_tensor(self.eps, tf.float32)

        self.tf_stft_istft_ins = TF_STFT_ISTFT(w=self.w, h=self.h)

        self.viterbi_ins = Viterbi(bins_per_semitone=self.bins_per_note, n_bins=self.U)

    def gen_WF0_fn(self):

        w = self.w

        wf0_ins = GFMSpec(fs=self.fs, n_samples=w, Oq=self.Oq, constantE=self.gfm_constant_E)

        WF0 = []
        for f0 in self.f0s:
            wf0 = wf0_ins.stft_of_f0_fn(f0)
            WF0.append(wf0)

        WF0 = np.stack(WF0, axis=-1)
        assert WF0.shape == (w // 2 + 1, self.U)

        WF0 = WF0 / np.max(WF0, axis=0)[None, :]

        WF0 = WF0.astype(np.float32)
        WF0.flags['WRITEABLE'] = False

        return WF0

    @staticmethod
    def tf_parameter_ini_fn(r, c):

        aa = tf.random.normal([r, c])
        aa = tf.abs(aa)

        return aa

    def ISD_fn(self, X, Y):

        t = (X + self.tf_eps) / (Y + self.tf_eps)
        t = -tf.math.log(t) + t
        t = tf.reduce_mean(t) - 1.

        return t

    def tf_imm_fn(self, SX):

        P = self.P
        R = self.R
        K = self.K
        U = self.U
        F = self.F
        n_iters = self.niters
        patient_iters = self.patient_iters

        eps = self.tf_eps
        WGAMMA = self.tf_WGAMMA
        WGAMMAT = self.tf_WGAMMAT
        WF0 = self.tf_WF0
        WF0T = self.tf_WF0T

        assert isinstance(SX, tf.Tensor)
        assert SX.shape[1] == F
        SX = tf.transpose(SX)
        N = tf.shape(SX)[1]

        # initialization
        HGAMMA = tf.abs(tf.random.normal([P, K]))
        HPHI = tf.abs(tf.random.normal([K, N]))
        HF0 = tf.abs(tf.random.normal([U, N]))
        WM = tf.abs(tf.random.normal([F, R]))
        HM = tf.abs(tf.random.normal([R, N]))

        # pre-processing
        WPHI = WGAMMA @ HGAMMA
        SPHI = WPHI @ HPHI
        SF0 = WF0 @ HF0
        SV = SPHI * SF0
        SM = WM @ HM
        hatSX = SV + SM

        min_err = None
        iters_since_last_update = 0

        for n_iter in range(n_iters):

            # HF0
            PSX = SPHI * SX / (hatSX ** 2 + eps)
            QSX = SPHI / (hatSX + eps)
            PSX = WF0T @ PSX
            QSX = WF0T @ QSX
            HF0 = HF0 * PSX / (QSX + eps)

            SF0 = WF0 @ HF0
            SV = SPHI * SF0
            hatSX = SV + SM

            # HPHI
            PSX = SF0 * SX / (hatSX ** 2 + eps)
            QSX = SF0 / (hatSX + eps)
            WPHIT = tf.transpose(WPHI)
            PSX = WPHIT @ PSX
            QSX = WPHIT @ QSX
            HPHI = HPHI * PSX / (QSX + eps)

            norm = tf.reduce_sum(HPHI, axis=0)
            HPHI = HPHI / (norm + eps)[None, :]
            HF0 = HF0 * norm[None, :]

            SPHI = WPHI @ HPHI
            SF0 = WF0 @ HF0
            SV = SPHI * SF0
            hatSX = SV + SM

            # HM
            PSX = SX / (hatSX ** 2 + eps)
            QSX = 1. / (hatSX + eps)
            WMT = tf.transpose(WM)
            PSX = WMT @ PSX
            QSX = WMT @ QSX
            HM = HM * PSX / (QSX + eps)

            SM = WM @ HM
            hatSX = SV + SM

            # HGAMMA
            PSX = SF0 * SX / (hatSX ** 2 + eps)
            QSX = SF0 / (hatSX + eps)
            HPHIT = tf.transpose(HPHI)
            PSX = WGAMMAT @ PSX @ HPHIT
            QSX = WGAMMAT @ QSX @ HPHIT
            HGAMMA = HGAMMA * PSX / (QSX + eps)

            norm = tf.reduce_sum(HGAMMA, axis=0)
            HGAMMA = HGAMMA / (norm + eps)[None, :]
            HPHI = HPHI * norm[:, None]
            norm = tf.reduce_sum(HPHI, axis=0)
            HPHI = HPHI / (norm + eps)[None, :]
            HF0 = HF0 * norm[None, :]

            WPHI = WGAMMA @ HGAMMA
            SPHI = WPHI @ HPHI
            SF0 = WF0 @ HF0
            SV = SPHI * SF0
            hatSX = SV + SM

            # WM
            PSX = SX / (hatSX ** 2 + eps)
            QSX = 1. / (hatSX + eps)
            HMT = tf.transpose(HM)
            PSX = PSX @ HMT
            QSX = QSX @ HMT
            WM = WM * PSX / (QSX + eps)

            norm = tf.reduce_sum(WM, axis=0)
            WM = WM / (norm + eps)[None, :]
            HM = HM * norm[:, None]

            SM = WM @ HM
            hatSX = SV + SM

            err = self.ISD_fn(SX, hatSX)

            logging.debug('{}/{} - {}'.format(n_iter, n_iters, err))


            if min_err is None or err < min_err:
                min_err = err
                best_result = dict(
                    HGAMMA=HGAMMA,
                    HPHI=HPHI,
                    HF0=HF0,
                    WM=WM,
                    HM=HM,
                    WPHI=WPHI,
                    SPHI=SPHI,
                    SF0=SF0,
                    SV=SV,
                    SM=SM,
                    hatSX=hatSX
                )
                iters_since_last_update = 0
            else:
                iters_since_last_update = iters_since_last_update + 1

            if iters_since_last_update == patient_iters:
                logging.info('no update within patient iters ({}), terminated earlier'.format(patient_iters))
                break

        else:
            logging.info('maximum number of iterations reached')

        return best_result

    def tf_stereo_imm_fn(self, SXL, SXR, sHF0):


        def WMb2(WM, b2):

            return WM * b2[None, :]

        def b2HM(b2, HM):

            return b2[:, None] * HM


        P = self.P
        R = self.R
        K = self.K
        F = self.F
        U = self.U
        n_iters = self.niters

        eps = self.tf_eps
        WGAMMA = self.tf_WGAMMA
        WGAMMAT = self.tf_WGAMMAT
        WF0 = self.tf_WF0
        WF0T = self.tf_WF0T

        assert isinstance(SXL, tf.Tensor)
        assert SXL.shape[1] == self.F
        SXL = tf.transpose(SXL)
        N = SXL.shape[1]

        assert isinstance(SXR, tf.Tensor)
        assert SXR.shape[1] == self.F
        SXR = tf.transpose(SXR)

        # initialization
        HGAMMA = tf.abs(tf.random.normal([P, K]))
        HPHI = tf.abs(tf.random.normal([K, N]))
        assert sHF0.shape == (U, N)
        HF0 = tf.convert_to_tensor(sHF0, tf.float32)
        WM = tf.abs(tf.random.normal([F, R]))
        HM = tf.abs(tf.random.normal([R, N]))

        alphaL = tf.convert_to_tensor(.5, tf.float32)
        alphaR = tf.convert_to_tensor(.5, tf.float32)

        betaL = tf.random.uniform([R])
        betaR = 1. - betaL
        betaL2 = betaL ** 2
        betaR2 = betaR ** 2

        WPHI = WGAMMA @ HGAMMA
        SPHI = WPHI @ HPHI
        SPHIL = alphaL ** 2 * SPHI
        SPHIR = alphaR ** 2 * SPHI
        SF0 = WF0 @ HF0
        SVL = SPHIL * SF0
        SVR = SPHIR * SF0
        SML = WMb2(WM, betaL2) @ HM
        SMR = WMb2(WM, betaR2) @ HM
        hatSXL = SVL + SML
        hatSXR = SVR + SMR

        patient_iters = self.patient_iters
        min_err = None
        iters_since_last_update = 0

        for n_iter in range(n_iters):

            # HF0
            PSXL = SPHIL * SXL / (hatSXL ** 2 + eps)
            QSXL = SPHIL / (hatSXL + eps)
            PSXR = SPHIR * SXR / (hatSXR ** 2 + eps)
            QSXR = SPHIR / (hatSXR + eps)
            PSX = PSXL + PSXR
            QSX = QSXL + QSXR
            PSX = WF0T @ PSX
            QSX = WF0T @ QSX
            HF0 = HF0 * PSX / (QSX + eps)

            SF0 = WF0 @ HF0
            SF0L = alphaL ** 2 * SF0
            SF0R = alphaR ** 2 * SF0
            hatSXL = SPHI * SF0L + SML
            hatSXR = SPHI * SF0R + SMR
            WPHIT = tf.transpose(WPHI)

            # HPHI
            PSXL = SF0L * SXL / (hatSXL ** 2 + eps)
            QSXL = SF0L / (hatSXL + eps)
            PSXR = SF0R * SXR / (hatSXR ** 2 + eps)
            QSXR = SF0R / (hatSXR + eps)
            PSX = PSXL + PSXR
            QSX = QSXL + QSXR
            PSX = WPHIT @ PSX
            QSX = WPHIT @ QSX
            HPHI = HPHI * PSX / (QSX + eps)

            norm = tf.reduce_sum(HPHI, axis=0)
            HPHI = HPHI / (norm + eps)[None, :]
            HF0 = HF0 * norm[None, :]

            SPHI = WPHI @ HPHI
            SF0 = WF0 @ HF0
            SF0L = alphaL ** 2 * SF0
            SF0R = alphaR ** 2 * SF0
            SVL = SPHI * SF0L
            SVR = SPHI * SF0R
            hatSXL = SVL + SML
            hatSXR = SVR + SMR

            WML = WMb2(WM, betaL2)
            WMR = WMb2(WM, betaR2)
            WMLT = tf.transpose(WML)
            WMRT = tf.transpose(WMR)

            # HM
            PSXL = SXL / (hatSXL ** 2 + eps)
            QSXL = 1. / (hatSXL + eps)
            PSXR = SXR / (hatSXR ** 2 + eps)
            QSXR = 1. / (hatSXR + eps)
            PSX = WMLT @ PSXL + WMRT @ PSXR
            QSX = WMLT @ QSXL + WMRT @ QSXR
            HM = HM * PSX / (QSX + eps)

            HML = b2HM(betaL2, HM)
            HMR = b2HM(betaR2, HM)
            SML = WM @ HML
            SMR = WM @ HMR
            hatSXL = SVL + SML
            hatSXR = SVR + SMR
            HPHIT = tf.transpose(HPHI)

            # HGAMMA
            PSXL = SF0L * SXL / (hatSXL ** 2 + eps)
            QSXL = SF0L / (hatSXL + eps)
            PSXR = SF0R * SXR / (hatSXR ** 2 + eps)
            QSXR = SF0R / (hatSXR + eps)
            PSX = PSXL + PSXR
            QSX = QSXL + QSXR
            PSX = WGAMMAT @ PSX @ HPHIT
            QSX = WGAMMAT @ QSX @ HPHIT
            HGAMMA = HGAMMA * PSX / (QSX + eps)

            norm = tf.reduce_sum(HGAMMA, axis=0)
            HGAMMA = HGAMMA / (norm + eps)[None, :]
            HPHI = HPHI * norm[:, None]
            norm = tf.reduce_sum(HPHI, axis=0)
            HPHI = HPHI / (norm + eps)[None, :]
            HF0 = HF0 * norm[None, :]

            WPHI = WGAMMA @ HGAMMA  # WPHI last updated
            SPHI = WPHI @ HPHI
            SF0 = WF0 @ HF0
            SV = SPHI * SF0
            SVL = alphaL ** 2 * SV
            SVR = alphaR ** 2 * SV
            hatSXL = SVL + SML
            hatSXR = SVR + SMR

            # WM
            HMLT = tf.transpose(HML)
            HMRT = tf.transpose(HMR)

            PSXL = SXL / (hatSXL ** 2 + eps)
            QSXL = 1. / (hatSXL + eps)
            PSXR = SXR / (hatSXR ** 2 + eps)
            QSXR = 1. / (hatSXR + eps)
            PSX = PSXL @ HMLT + PSXR @ HMRT
            QSX = QSXL @ HMLT + QSXR @ HMRT
            WM = WM * PSX / (QSX + eps)

            norm = tf.reduce_sum(WM, axis=0)
            WM = WM / (norm + eps)[None, :]
            HM = HM * norm[:, None]

            SML = WMb2(WM, betaL2) @ HM
            SMR = WMb2(WM, betaR2) @ HM
            hatSXL = SVL + SML
            hatSXR = SVR + SMR

            # alpha
            PSXL = SV * SXL / (hatSXL ** 2 + eps)
            QSXL = SV / (hatSXL + eps)
            PSXL = tf.reduce_sum(PSXL)
            QSXL = tf.reduce_sum(QSXL)
            alphaL = alphaL * (PSXL / (QSXL + eps)) ** .1

            PSXR = SV * SXR / (hatSXR ** 2 + eps)
            QSXR = SV / (hatSXR + eps)
            PSXR = tf.reduce_sum(PSXR)
            QSXR = tf.reduce_sum(QSXR)
            alphaR = alphaR * (PSXR / (QSXR + eps)) ** .1

            alphaL = alphaL + eps
            alphaR = alphaR + eps
            alphaL = alphaL / (alphaL + alphaR)
            alphaR = 1. - alphaL

            hatSXL = alphaL ** 2 * SV + SML
            hatSXR = alphaR ** 2 * SV + SMR

            # beta
            WMT = tf.transpose(WM)

            PSXL = SXL / (hatSXL ** 2 + eps)
            QSXL = 1. / (hatSXL + eps)
            PSXL = tf.reduce_sum((WMT @ PSXL) * HM, axis=1)
            QSXL = tf.reduce_sum((WMT @ QSXL) * HM, axis=1)
            betaL = betaL * (PSXL / (QSXL + eps)) ** .1

            PSXR = SXR / (hatSXR ** 2 + eps)
            QSXR = 1. / (hatSXR + eps)
            PSXR = tf.reduce_sum((WMT @ PSXR) * HM, axis=1)
            QSXR = tf.reduce_sum((WMT @ QSXR) * HM, axis=1)
            betaR = betaR * (PSXR / (QSXR + eps)) ** .1

            betaL = betaL + eps
            betaR = betaR + eps
            betaL = betaL / (betaL + betaR)
            betaR = 1. - betaL
            betaL2 = betaL ** 2
            betaR2 = betaR ** 2

            SPHIL = alphaL ** 2 * SPHI
            SPHIR = alphaR ** 2 * SPHI
            SVL = SPHIL * SF0
            SVR = SPHIR * SF0
            SML = WMb2(WM, betaL2) @ HM  # SML last updated
            SMR = WMb2(WM, betaR2) @ HM
            hatSXL = SVL + SML
            hatSXR = SVR + SMR

            lossL = self.ISD_fn(SXL, hatSXL)
            lossR = self.ISD_fn(SXR, hatSXR)
            logging.debug('{}/{} - {}, {}'.format(n_iter, n_iters, lossL.numpy(), lossR.numpy()))

            err = (lossL + lossR) / 2.
            if min_err is None or err < min_err:
                min_err = err
                result = dict(
                    HGAMMA=HGAMMA,
                    HPHI=HPHI,
                    HF0=HF0,
                    alphaL=alphaL,
                    alphaR=alphaR,
                    betaL=betaL,
                    betaR=betaR,
                    SVL=SVL,
                    SVR=SVR,
                    SML=SML,
                    SMR=SMR,
                    hatSXL=hatSXL,
                    hatSXR=hatSXR
                )
                iters_since_last_update = 0
            else:
                iters_since_last_update = iters_since_last_update + 1

            if iters_since_last_update == patient_iters:
                logging.info('no update with patient iters {}, terminated earlier'.format(patient_iters))
                break
        else:
            logging.info('maximum number of iters reached')

        return result

    def output_melody_fn(self, melody_states, voicing):

        assert isinstance(melody_states, np.ndarray)
        assert isinstance(voicing, np.ndarray)
        assert len(melody_states) == len(voicing)

        U = self.U

        f0_table = self.f0s

        melody_states = np.minimum(melody_states, U - 1)
        f0s = f0_table[melody_states]
        f0s = np.where(voicing, f0s, 0.)

        return f0s

    def get_energies_for_f0s_fn(self, results_dict, SX):

        WF0 = self.tf_WF0
        HF0 = results_dict['HF0']
        SPHI = results_dict['SPHI']
        hatSX = results_dict['hatSX']
        assert SX.shape[1] == self.F
        SX = tf.transpose(SX)
        N = SX.shape[1]

        U = self.U
        hatSX = hatSX + self.eps
        energies_FN = np.empty([U, N], np.float32)
        for u in range(U):
            uSF0 = HF0[u, :][None, :] * WF0[:, u][:, None]  # F * N
            SV = SPHI * uSF0
            ratio = (SV / hatSX) ** 2
            SV = ratio * SX
            SV = tf.reduce_sum(SV, axis=0)
            energies_FN[u] = SV

        return energies_FN

    def logits_fn(self, wave_file_name):

        fs = self.fs

        x, sr = librosa.load(wave_file_name, sr=self.fs, mono=True)
        assert sr == int(fs)
        X = self.tf_stft_istft_ins.tf_stft_fn(x)
        SX = tf.abs(X) ** 2

        melody_imm_results_dict = self.tf_imm_fn(SX)

        energies = self.get_energies_for_f0s_fn(melody_imm_results_dict, SX)
        hw = self.w // 2
        hw = hw ** 2
        np.divide(energies, float(hw), out=energies)
        np.maximum(energies, 1e-11, out=energies)
        np.log10(energies, out=energies)
        np.add(energies, 6., out=energies)

        return energies

    def melody_fn(self, wave_file_name):

        fs = self.fs

        x, sr = librosa.load(wave_file_name, sr=self.fs, mono=True)
        assert sr == int(fs)
        X = self.tf_stft_istft_ins.tf_stft_fn(x)
        SX = tf.abs(X) ** 2

        melody_imm_results_dict = self.tf_imm_fn(SX)

        HF0 = melody_imm_results_dict['HF0']
        melody_states = self.viterbi_ins(HF0)

        voicing = self.voicing_detection_fn(
            tf_SX=SX,
            melody_imm_results_dict=melody_imm_results_dict,
            melody_states=melody_states
        )

        return dict(
            voicing=voicing,
            bins=np.minimum(melody_states, self.U - 1)
        )

    def voicing_detection_fn(self, tf_SX, melody_imm_results_dict, melody_states):


        HF0 = melody_imm_results_dict['HF0'].numpy()
        HF0 = np.require(HF0, requirements=['F'])

        assert isinstance(melody_states, np.ndarray)
        assert melody_states.dtype == np.int64

        U = self.U
        assert HF0.shape[0] == U
        N = HF0.shape[1]
        assert N == len(melody_states)

        tf_melody_states = tf.convert_to_tensor(melody_states)
        frames_voiced = tf_melody_states < U
        frames_voiced = frames_voiced.numpy()
        offset = self.bins_per_note // 2

        start_bins = tf_melody_states - offset
        end_bins = tf_melody_states + offset + 1
        start_bins = tf.maximum(start_bins, 0)
        end_bins = tf.minimum(end_bins, U)
        start_bins = start_bins.numpy()
        end_bins = end_bins.numpy()

        sHF0 = np.zeros_like(HF0)

        for frame_idx, voiced in enumerate(frames_voiced):

            if voiced:
                start_bin = start_bins[frame_idx]
                end_bin = end_bins[frame_idx]
                if start_bin < end_bin:
                    sHF0[start_bin:end_bin, frame_idx] = HF0[start_bin:end_bin, frame_idx]

        SF0 = self.tf_WF0 @ sHF0
        SV = melody_imm_results_dict['SPHI'] * SF0
        hatSX = SV + melody_imm_results_dict['SM']
        t = (SV + self.tf_eps) / (hatSX + self.tf_eps)
        t = t ** 2 * tf.transpose(tf_SX)
        frame_energies = tf.reduce_sum(t, axis=0)
        frame_energies_sorted = tf.sort(frame_energies)
        t = tf.cumsum(frame_energies_sorted)
        t = t / t[-1]
        energy_threshold = 5.84e-4
        t = t > energy_threshold
        idx = tf.argmax(t, output_type=tf.int32)
        energy_threshold = frame_energies_sorted[idx]
        frames_voiced = frame_energies > energy_threshold
        frames_voiced = frames_voiced.numpy()

        return frames_voiced

    def save_stereo_wav_fn(self, xL, xR, n_samples, wav_file_name):

        if isinstance(xL, tf.Tensor):
            xL = xL.numpy()
            xR = xR.numpy()

        xL = xL[:n_samples]
        xR = xR[:n_samples]
        c = 32767
        xL = np.clip(xL, -1., 1.)
        xL = xL * c
        xL = xL.astype(np.int16)
        xR = np.clip(xR, -1., 1.)
        xR = xR * c
        xR = xR.astype(np.int16)
        x = np.stack([xL, xR], axis=1)

        soundfile.write(wav_file_name, data=x, samplerate=int(self.fs))


if __name__ == '__main__':

    test_songlist = ["AClassicEducation_NightOwl", "Auctioneer_OurFutureFaces", "CelestialShore_DieForUs",
                     "Creepoid_OldTree", "Debussy_LenfantProdigue", "MatthewEntwistle_DontYouEver",
                     "MatthewEntwistle_Lontano", "Mozart_BesterJungling", "MusicDelta_Gospel",
                     "PortStWillow_StayEven", "Schubert_Erstarrung", "StrandOfOaks_Spacestation"]

    track_id = test_songlist[2]

    wav_file = os.path.join(os.environ['medleydb'], track_id, track_id + '_MIX.wav')

    imm_ins = IMM()
    energies = imm_ins.SIMM_fn(wav_file)




    librosa.display.specshow(
            librosa.power_to_db(data, ref=np.max),
            x_axis='time',
            y_axis='cqt_hz',
            sr=imm_ins.fs,
            hop_length=imm_ins.h,
            fmin=imm_ins.fmin,
            bins_per_octave=imm_ins.bins_per_note * 12
    )
    plt.savefig('{}_log.png'.format(track_id))
    plt.close()

































