from ...config.config import model_config
from ..layers.gaussian_process import GaussianProcessRegressor

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Attention
from tensorflow.keras.models import Model



class DeepFactor(Model):
    def __init__(self, config=model_config, **kwargs):
        super(DeepFactor, self).__init__(**kwargs)
        self.config = config

        self.gaussian_process = GaussianProcessRegressor()
        self.lstm = LSTM(units=config.lstm_units, return_sequences=True)
        self.dense = Dense(units=1)
        self.attention = Attention()

    def call(self, inputs, training=None):
        # fixed effect     
        _g_it = [self.lstm(inputs) for _ in range(self.config.K)] # [batch_size, time_seq, lstm_units] * 10
        _g_it = tf.convert_to_tensor([self.dense(i) for i in _g_it])  # n_factors, batch_size, time_seq, 1
        _g_it = tf.squeeze(_g_it) # n_factors, batch_size, time_seq
        g_it = tf.transpose(_g_it, (1, 2, 0)) # batch_size, time_seq, n_factors

        # random effect
        _r_i = tf.convert_to_tensor([[self.gaussian_process(inputs[i, :, j])[0] for i in range(inputs.shape[0])] for j in range(inputs.shape[-1])])
        r_it = tf.transpose(_r_i, perm=(1, 2, 0))
        
        attention_scores = tf.nn.softmax(Attention()([g_it, g_it]), axis=-1)
        f_it = attention_scores * g_it

        # emission
        u_it = f_it + r_it
        _z_i = tf.convert_to_tensor([[self.gaussian_process(u_it[i, :, j])[0] for i in range(u_it.shape[0])] for j in range(u_it.shape[-1])])
        z_it = tf.transpose(_z_i, perm=(1, 2, 0))

        return z_it
