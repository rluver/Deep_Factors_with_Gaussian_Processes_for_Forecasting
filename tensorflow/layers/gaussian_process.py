from ..kernels.kernels import SquaredExponentialKernel

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


class GaussianProcessRegressor(Model):
    def __init__(self, sigma=0.5, kernel=SquaredExponentialKernel()):
        super(GaussianProcessRegressor, self).__init__()
        self.sigma = sigma
        self.kernel = kernel        

        self._calculate_covariance_matrix = lambda x1, x2: self.kernel(x1, x2)

    def build(self, input_shape):
        self.n = input_shape[0]
        self.identity = tf.eye(input_shape[0])

    def call(self, inputs, targets):
        x = tf.broadcast_to(inputs, shape=(inputs.shape[0], inputs.shape[0]))
        y = tf.broadcast_to(targets, shape=(targets.shape[0], targets.shape[0]))
        
        K = self._calculate_covariance_matrix(x, tf.transpose(x))
        K_star = self._calculate_covariance_matrix(x, tf.transpose(y))
        K_star2 = self._calculate_covariance_matrix(y, tf.transpose(y))

        cov_y = K + (self.sigma**2)*self.identity
        
        L = tf.linalg.cholesky(cov_y)
        alpha = tf.transpose(L) / (L/targets)
        f_star = (tf.transpose(K_star) @ alpha)[:, 0]
        v = L / K_star
        V_f_star = K_star2 - tf.transpose(v) @ v

        _targets = targets[tf.newaxis, :]
        log_marginal_likelihood = (
            -0.5 * tf.squeeze(_targets@tf.linalg.inv(cov_y)@tf.transpose(_targets)) 
            -0.5 * tf.math.log(tf.linalg.det(cov_y)) 
            -self.n/2 * tf.math.log(2*np.pi)
        )

        return f_star, V_f_star, log_marginal_likelihood
