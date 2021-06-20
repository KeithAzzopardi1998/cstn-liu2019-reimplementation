import numpy as np
from keras import backend as K
import tensorflow as tf

class Metrics():
    def __init__(self,odmax):
        self.odmax=odmax

    def mean_squared_error(self,y_true, y_pred):
        return K.mean(K.square(y_pred - y_true))

    def rmse(self,y_true, y_pred):
        return self.mean_squared_error(y_true, y_pred) ** 0.5

    def mape(self,y_true, y_pred):
        y_true = (y_true + 1) * self.odmax / 2.0
        y_pred = (y_pred + 1) * self.odmax / 2.0
        y_pred = tf.round(y_pred)
        mask = tf.greater_equal(y_true, K.constant(5.0))
        return tf.reduce_mean(tf.boolean_mask((tf.abs(y_true - y_pred)/y_true), mask))


    def o_rmse(self,y_true, y_pred):
        y_true = K.sum(y_true, axis=1)
        y_pred = K.sum(y_pred, axis=1)
        return self.mean_squared_error(y_true, y_pred) ** 0.5

    def o_mape(self,y_true, y_pred):
        y_true = (y_true + 1) * self.odmax / 2.0
        y_pred = (y_pred + 1) * self.odmax / 2.0
        y_true = K.sum(y_true, axis=1)
        y_pred = K.sum(y_pred, axis=1)
        y_pred = tf.round(y_pred)
        mask = tf.greater_equal(y_true, K.constant(5.0))
        return tf.reduce_mean(tf.boolean_mask((tf.abs(y_true - y_pred)/y_true), mask))

    def shape1(self,y_true, y_pred):
        # batch
        return K.shape(y_pred)[0]


    def shape2(self,y_true, y_pred):
        return K.shape(y_pred)[1]


    def shape3(self,y_true, y_pred):
        return K.shape(y_pred)[2]


    def shape4(self,y_true, y_pred):
        return K.shape(y_pred)[3]
