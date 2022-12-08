import tensorflow as tf


class Embedding(tf.keras.layers.Layer):
    def __init__(self, units, activations):
        super(Embedding, self).__init__()
        assert len(units) == len(activations), "length of units and activations must be the same"
        self.units = units
        self.activations = activations

        self.weights_ = None
        self.biases_ = None

    def build(self, input_shape):
        self.weights_ = []
        self.biases_ = []

        for i, units in enumerate(self.units):
            w_init = tf.random_normal_initializer()(shape=(input_shape[-1], units), dtype='float32')
            weight = tf.Variable(name="w%i" % (i + 1), initial_value=w_init, trainable=True)

            b_init = tf.zeros_initializer()(shape=(units,), dtype='float32')
            bias = tf.Variable(name="b%i" % (i + 1), initial_value=b_init, trainable=True)
            self.weights_.append(weight)
            self.biases_.append(bias)

        super().build(input_shape)

    def call(self, inputs):
        res = inputs
        for i, activation_name in enumerate(self.activations):
            activation = tf.keras.activations.get(activation_name)
            res = activation(tf.matmul(inputs, self.weights_[i]) + self.biases_[i])
        return res


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units, activations, num_vars=60):
        super(AttentionLayer, self).__init__()
        self.units = units
        self.activations = activations
        self.num_vars = num_vars

        self.cons_embedding = None
        self.cuts_embedding = None

    def build(self, input_shape):
        self.cons_embedding = Embedding(self.units, self.activations)
        self.cons_embedding.build(input_shape)

        self.cuts_embedding = Embedding(self.units, self.activations)
        self.cuts_embedding.build(input_shape)

        super().build(input_shape)

    def call(self, inputs):
        """
        inputs: list of constraints array and Gomory cuts array
        """
        cons = inputs[0]
        cuts = inputs[1]

        h = self.cons_embedding(cons)
        g = self.cuts_embedding(cuts)

        S = tf.reduce_mean(tf.matmul(g, tf.transpose(h)), axis=1)
        return tf.nn.softmax(S)


class Policy:
    def __init__(self, units, activations, lr, num_vars=60):
        self.units = units
        self.activations = activations
        self.num_vars = num_vars

        self.cons_embedding = Embedding(self.units, self.activations)
        self.cuts_embedding = Embedding(self.units, self.activations)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def compute_prob(self, states):
        cons = states[0]
        cuts = states[1]

        h = self.cons_embedding(cons)
        g = self.cuts_embedding(cuts)

        S = tf.reduce_mean(tf.matmul(g, tf.transpose(h)), axis=1)
        return tf.cast(tf.nn.softmax(S), tf.double).numpy()

    def train(self, states, actions, Qs):
        with tf.GradientTape() as tape:
            prob = self.compute_prob(states)




