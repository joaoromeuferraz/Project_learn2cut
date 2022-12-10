import tensorflow as tf


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

