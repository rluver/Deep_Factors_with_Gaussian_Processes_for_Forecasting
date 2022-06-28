from ..kernels.kernels import SquaredExponentialKernel

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class GaussianProcessRegressor(Layer):
    def __init__(self, sigma=0.5, kernel=SquaredExponentialKernel()):
        super(GaussianProcessRegressor, self).__init__()
        self.sigma = sigma
        self.kernel = kernel        

        self._calculate_covariance_matrix = lambda x1, x2: self.kernel(x1, x2)

    def build(self, input_shape):
        self.n = input_shape[0]
        self.identity = tf.eye(input_shape[0])

    def call(self, inputs, training=None):
        x = tf.broadcast_to(inputs, shape=(inputs.shape[0], inputs.shape[0]))
        
        K = self._calculate_covariance_matrix(x, tf.transpose(x))

        cov_y = K + (self.sigma**2)*self.identity
        
        L = tf.linalg.cholesky(cov_y)
        alpha = tf.transpose(L) / (L/inputs)
        f_star = (tf.transpose(K) @ alpha)[:, 0]
        v = L / K
        V_f_star = K - tf.transpose(v) @ v

        _inputs = inputs[tf.newaxis, :]
        log_marginal_likelihood = (
            -0.5 * tf.squeeze(_inputs@tf.linalg.inv(cov_y)@tf.transpose(_inputs)) 
            -0.5 * tf.math.log(tf.linalg.det(cov_y)) 
            -self.n/2 * tf.math.log(2*np.pi)
        )

        return f_star, V_f_star, log_marginal_likelihood
