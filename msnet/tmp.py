import tensorflow as tf
import numpy as np

aa = tf.convert_to_tensor([1, 2, 4, 5, 6, 0, 9])
indices = [3, 4, 0, 1]
bb = tf.gather(aa, indices)
print(bb.numpy())