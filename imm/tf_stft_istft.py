import os

import librosa
import tensorflow as tf
import numpy as np


class TF_STFT_ISTFT:

    def __init__(self, w, h):

        assert w % h == 0

        window = TF_STFT_ISTFT.sinebell_fn(w)
        self.overlap = np.sum(window ** 2) / w * (w // h)
        window = window.astype(np.float32)
        self.window = window

        self.w = w
        self.h = h

    @staticmethod
    def sinebell_fn(w):

        window = np.sin(np.pi * np.arange(w) / w)

        return window

    @tf.function(input_signature=[tf.TensorSpec([None], name='y')])
    def tf_stft_fn(self, y):

        h = tf.convert_to_tensor(self.h, tf.int32)
        w = tf.convert_to_tensor(self.w, tf.int32)

        y = tf.convert_to_tensor(y, tf.float32)
        y.set_shape([None])

        n_samples = tf.size(y, out_type=tf.int32)

        n_frames = (n_samples + h - 1) // h
        left_padding = w // 2
        required_n_samples = (n_frames - 1) * h + w
        right_padding = required_n_samples - (n_samples + left_padding)
        r = w - left_padding - 1
        min_right_padding = r - (h - 1)
        tf.debugging.assert_greater_equal(right_padding, min_right_padding)
        tf.debugging.assert_less_equal(right_padding, r)

        y = tf.pad(y, [[left_padding, right_padding]], mode='reflect')
        _n_samples = tf.size(y, out_type=tf.int32)
        tf.debugging.assert_equal(_n_samples, required_n_samples)

        y = tf.signal.frame(
            y,
            frame_length=w,
            frame_step=h,
            pad_end=False
        )
        shape = tf.shape(y, out_type=tf.int32)
        tf.debugging.assert_equal(shape, [n_frames, w])

        window = tf.convert_to_tensor(self.window, tf.float32)
        y = y * window[None, :]
        y = tf.signal.rfft(y, fft_length=[self.w])
        tf.debugging.assert_equal(tf.shape(y), [n_frames, w // 2 + 1])

        return y

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.complex64, name='y')])
    def tf_istft_fn(self, y):

        w = tf.convert_to_tensor(self.w, tf.int32)
        h = tf.convert_to_tensor(self.h, tf.int32)
        window = tf.convert_to_tensor(self.window, tf.float32)
        scale = tf.convert_to_tensor(1. / self.overlap, tf.float32)

        y = tf.convert_to_tensor(y, tf.complex64)
        y.set_shape([None, self.w // 2 + 1])
        n_frames = tf.shape(y)[0]
        n_samples = (n_frames - 1) * h + w

        y = tf.signal.irfft(y, fft_length=[self.w])
        y = y * window[None, :]
        y = tf.signal.overlap_and_add(
            y, frame_step=h
        )
        tf.debugging.assert_equal(tf.size(y), n_samples)
        y = y[w // 2:]
        y = y * scale

        return y


if __name__ == '__main__':

    samples, _ = librosa.load(os.environ['wav_file_short'], sr=None)

    w = 2048
    h = 256
    tf_ins = TF_STFT_ISTFT(w=w, h=h)
    _samples = tf_ins.tf_stft_fn(samples)
    _samples = tf_ins.tf_istft_fn(_samples).numpy()
    n_samples = len(samples)
    assert len(_samples) >= n_samples
    _samples = _samples[:n_samples]
    samples = samples[w:-w]
    _samples = _samples[w:-w]
    d = np.abs(samples - _samples)
    t = np.sum(d)
    print(d.max(), t)








