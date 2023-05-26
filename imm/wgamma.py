import numpy as np


def gen_WGAMMA_fn(n_freq_bins, n_bases, overlap):

    assert overlap > 0
    assert overlap < 1

    O = overlap
    Ob = 1. - O
    n = int(np.ceil(1. / Ob))
    Ob = 1. / n
    O = 1. - Ob

    w = float(n_freq_bins) / ((n_bases - 1) * Ob + 1 - 2 * O)
    w = int(np.ceil(w))
    if w % 2 != 0:
        w = w - 1

    centers = np.arange(n_bases) * Ob + (Ob - O) / 2.
    centers = centers * w
    centers = centers.astype(np.int32)

    WGAMMA = np.zeros([n_freq_bins, n_bases])
    hL = w // 2
    window = np.hanning(w)  # symmetric

    for p in range(n_bases):
        c = centers[p]
        s = c - hL
        e = c + hL
        for real_p, hann_p in zip(range(s, e), range(w)):
            if real_p < 0 or real_p >= n_freq_bins:
                continue

            WGAMMA[real_p, p] = window[hann_p]

    WGAMMA = WGAMMA.astype(np.float32)
    WGAMMA.flags['WRITEABLE'] = False

    return WGAMMA


def generateHannBasis(numberFrequencyBins, sizeOfFourier, Fs, \
                      frequencyScale='linear', numberOfBasis=20, \
                      overlap=.75):
    isScaleRecognized = False
    if frequencyScale == 'linear':
        # number of windows generated:
        numberOfWindowsForUnit = np.ceil(1.0 / (1.0 - overlap))
        # recomputing the overlap to exactly fit the entire
        # number of windows:
        overlap = 1.0 - 1.0 / np.double(numberOfWindowsForUnit)
        # length of the sine window - that is also to say: bandwidth
        # of the sine window:
        lengthSineWindow = numberFrequencyBins / ((1. - overlap) * (numberOfBasis - 1) + 1 - 2. * overlap)
        lengthSineWindow = np.ceil(lengthSineWindow)

        # even window length, for convenience:
        lengthSineWindow = 2.0 * np.floor(lengthSineWindow / 2.0)

        # for later compatibility with other frequency scales:
        mappingFrequency = np.arange(numberFrequencyBins)

        # size of the "big" window
        sizeBigWindow = 2.0 * numberFrequencyBins

        O = overlap
        Ob = 1. - O
        t = np.arange(numberOfBasis) * Ob + (Ob - O) / 2.
        t = t * lengthSineWindow
        sineCenters = t

        # For future purpose: to use different frequency scales
        isScaleRecognized = True

    # For frequency scale in logarithm (such as ERB scales)
    if frequencyScale == 'log':
        isScaleRecognized = False

    # checking whether the required scale is recognized
    if not(isScaleRecognized):
        print("The desired feature for frequencyScale is not recognized yet...")
        return 0

    # the shape of one window:
    prototypeSineWindow = np.hanning(lengthSineWindow)
    # adding zeroes on both sides, such that we do not need to check
    # for boundaries
    sizeBigWindow = int(sizeBigWindow)
    lengthSineWindow = int(lengthSineWindow)
    bigWindow = np.zeros([sizeBigWindow * 2, 1])
    bigWindow[(sizeBigWindow - lengthSineWindow // 2): \
              (sizeBigWindow + lengthSineWindow // 2)] \
        = np.vstack(prototypeSineWindow)

    WGAMMA = np.zeros([numberFrequencyBins, numberOfBasis])
    W = np.zeros([numberFrequencyBins, numberOfBasis])
    hL = lengthSineWindow // 2
    for p in np.arange(numberOfBasis):
        WGAMMA[:, p] = np.hstack(bigWindow[mappingFrequency \
                                                    - int(sineCenters[p]) \
                                                    + sizeBigWindow])

        c = sineCenters[p]
        c = int(c)
        s = c - hL
        e = c + hL
        for real_p, hann_p in zip(range(s, e), range(lengthSineWindow)):
            if real_p < 0 or real_p >= numberFrequencyBins:
                continue
            W[real_p, p] = prototypeSineWindow[hann_p]
        t = np.sum(np.abs(WGAMMA[:, p] - W[:, p]))

    return WGAMMA


if __name__ == '__main__':

    n_freq_bins = 1500
    n_bases = 40
    overlap = 0.34

    w = gen_WGAMMA_fn(n_freq_bins=n_freq_bins, n_bases=n_bases, overlap=overlap)
    _w = generateHannBasis(
        numberFrequencyBins=n_freq_bins,
        sizeOfFourier=(n_freq_bins - 1) * 2,
        Fs=44100,
        frequencyScale='linear',
        numberOfBasis=n_bases,
        overlap=overlap)

    d = np.sum(np.abs(w - _w))
    print(d)


