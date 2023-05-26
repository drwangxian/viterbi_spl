import numpy as np


class GFMSpec:

    def __init__(self, fs, n_samples, Oq, constantE=True):

        assert Oq > 0
        assert Oq < 1

        self.fs = fs
        self.n_samples = n_samples
        self.Oq = Oq
        self.keep_E = constantE
        self.window = np.sin(np.pi * np.arange(n_samples) / n_samples)

    def stft_of_f0_fn(self, f0):

        Oq = self.Oq
        fs = self.fs
        n_samples = self.n_samples

        j2pi = 1j * 2 * np.pi

        n_hs = fs / 2. / f0
        n_hs = int(np.floor(n_hs))

        s = j2pi* np.arange(1, n_hs + 1) * Oq
        one_over_s = 1. / s
        ems = np.exp(-s)
        if self.keep_E:
            chs = Oq * one_over_s * (
                ems
                + 2. * (1. + 2. * ems) * one_over_s
                - 6. * (1. - ems) * one_over_s ** 2
            )
        else:
            chs = 27. / 4. * f0 * one_over_s * (
                    ems
                    + 2. * (1. + 2. * ems) * one_over_s
                    - 6. * (1. - ems) * one_over_s ** 2
            )

        ts = np.arange(n_samples)
        hf0t = np.arange(1, n_hs + 1) * (f0 / float(fs))
        hf0t = hf0t[:, None] * ts[None, :]
        assert hf0t.shape == (n_hs, n_samples)
        exp_real = np.cos(2 * np.pi * hf0t)
        exp_imag = np.sin(2 * np.pi * hf0t)
        ch_real = chs.real
        ch_imag = chs.imag
        _real = ch_real[:, None] * exp_real
        _imag = ch_imag[:, None] * exp_imag
        waveform = _real - _imag
        waveform = np.sum(waveform, axis=0)
        spec = np.fft.rfft(waveform * self.window)
        spec = np.abs(spec) ** 2

        return spec


def sinebell(lengthWindow):
    """
    window = sinebell(lengthWindow)

    Computes a "sinebell" window function of length L=lengthWindow

    The formula is:
        window(t) = sin(pi * t / L), t = 0..L-1

    sinebell = sqrt(hann)
    """

    lengthWindow = int(lengthWindow)

    window = np.sin(np.pi * np.arange(lengthWindow) / lengthWindow)

    return window


def generate_ODGD_spec(F0, Fs, lengthOdgd=2048, Nfft=2048, Ot=0.5, \
                       t0=0.0, analysisWindowType='sinebell'):
    """
    generateODGDspec:

    generates a waveform ODGD and the corresponding spectrum,
    using as analysis window the -optional- window given as
    argument.
    """

    j2pi = 1j * 2 * np.pi

    assert Ot > 0
    assert Ot < 1

    # converting input to double:
    F0 = float(F0)
    Fs = float(Fs)
    Ot = float(Ot)
    t0 = float(t0)

    # compute analysis window of given type:
    # if analysisWindowType=='sinebell':
    analysisWindow = sinebell(lengthOdgd)
    # else:
    #     if analysisWindowType=='hanning' or \
    #             analysisWindowType=='hanning':
    #         analysisWindow = hann(lengthOdgd)

    # maximum number of partials in the spectral comb:
    partialMax = int(np.floor((Fs / 2) / F0))

    # Frequency numbers of the partials:
    frequency_numbers = np.arange(1,partialMax + 1)
    s = j2pi * Ot * frequency_numbers
    one_over_s = 1. / s
    ems = np.exp(-s)
    amplitudes = 27./ 4. * F0 * one_over_s * (
            ems
            + 2 * (1. + 2 * ems) * one_over_s
            - 6 * (1. - ems) * one_over_s ** 2
    )

    # Time stamps for the time domain ODGD
    timeStamps = np.arange(lengthOdgd) / Fs + t0 / F0
    j2pihf0s = j2pi * F0 * frequency_numbers
    j2pihf0st = j2pihf0s[:, None] * timeStamps[None, :]
    exp_j2pihf0st = np.exp(j2pihf0st)
    exp_real = exp_j2pihf0st.real
    exp_img = exp_j2pihf0st.imag
    amp_real = amplitudes.real
    amp_img = amplitudes.imag
    waveform = exp_real * amp_real[:, None] - exp_img * amp_img[:, None]
    waveform = np.sum(waveform, axis=0)
    spec = np.fft.rfft(waveform * analysisWindow)
    spec = np.abs(spec) ** 2

    return spec


if __name__ == '__main__':

    spec_ins = GFMSpec(
        fs=44100,
        n_samples=2048,
        Oq=0.25,
        constantE=True
    )

    spec = spec_ins.stft_of_f0_fn(800)

    print()

