import tensorflow as tf
from tensorflow.keras.layers import Layer


class SquaredExponentialKernel(Layer):
    def __init__(self, sigma=1, l=1, **kwargs):
        super(SquaredExponentialKernel, self).__init__(**kwargs)
        self.sigma = sigma
        self.l = l

    def call(self, x1, x2):
        return (self.sigma ** 2) * tf.math.exp(-(x1 - x2)**2 / (2 * (self.l**2)))
