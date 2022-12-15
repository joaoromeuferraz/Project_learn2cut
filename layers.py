import tensorflow as tf
import numpy as np

class Embedding(tf.keras.layers.Layer):
    def __init__(self, units, activations):
        super(Embedding, self).__init__()
        assert len(units) == len(activations), "length of units and activations must be the same"
        self.units = units
        self.activations = activations

        self.w = None
        self.b = None

    def build(self, input_shape):
        self.w = []
        self.b = []

        prev_units = input_shape[-1]

        for i, units in enumerate(self.units):
            weight = self.add_weight(shape=(prev_units, units), initializer="random_normal", trainable=True)
            bias = self.add_weight(shape=(units, ), initializer="zeros", trainable=True)
            self.w.append(weight)
            self.b.append(bias)
            prev_units = units

        super().build(input_shape)

    def call(self, inputs):
        res = inputs
        for i, activation_name in enumerate(self.activations):
            activation = tf.keras.activations.get(activation_name)
            res = activation(tf.matmul(res, self.w[i]) + self.b[i])
        return res


class AttentionEmbedding(tf.keras.layers.Layer):
    def __init__(self, units, activations, num_vars=60):
        super(AttentionEmbedding, self).__init__()
        assert len(units) == len(activations), "length of units and activations must be the same"
        self.units = units
        self.activations = activations
        self.num_vars = num_vars

        self.w_cons = []
        self.b_cons = []
        
        self.w_cuts = []
        self.b_cuts = []

    def build(self, input_shape):
        self.w = []
        self.b = []
        
        prev_units = self.num_vars + 1

        for i, units in enumerate(self.units):
            w1 = self.add_weight(shape=(prev_units, units), initializer="random_normal", trainable=True)
            w2 = self.add_weight(shape=(prev_units, units), initializer="random_normal", trainable=True)
            b1 = self.add_weight(shape=(units, ), initializer="zeros", trainable=True)
            b2 = self.add_weight(shape=(units, ), initializer="zeros", trainable=True)
            
            self.w_cons.append(w1)
            self.b_cons.append(b1)
            
            self.w_cuts.append(w2)
            self.b_cuts.append(b2)
            
            prev_units = units

        super().build(input_shape)

    def call(self, inputs):
        cons = tf.cast(tf.concat((inputs[0], tf.expand_dims(inputs[1], 1)), axis=1), tf.float32)
        cuts = tf.cast(tf.concat((inputs[3], tf.expand_dims(inputs[4], 1)), axis=1), tf.float32)
        
        h = cons
        g = cuts
        
        for i, activation_name in enumerate(self.activations):
            activation = tf.keras.activations.get(activation_name)
            h = activation(tf.matmul(h, self.w_cons[i]) + self.b_cons[i])
            g = activation(tf.matmul(g, self.w_cuts[i]) + self.b_cuts[i])
            
        S = tf.reduce_mean(tf.matmul(g, tf.transpose(h)), axis=1)
        return S