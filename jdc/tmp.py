import librosa.util
import numpy as np
import tensorflow as tf

a = np.asarray([1., 2])
print(np.exp(a))
np.exp(a, out=a)
print(a)