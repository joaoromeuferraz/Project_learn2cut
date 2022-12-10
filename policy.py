import numpy as np
import tensorflow as tf
from layers import Embedding
from utils import pad_state


class BasePolicy(object):
    def __init__(self, policy_params):
        self.numvars = policy_params['num_vars']
        self.weights = np.empty(0)

    def update_weights(self, new_weights):
        self.weights[:] = new_weights[:]
        return

    def get_weights(self):
        return self.weights

    def act(self, ob):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def reset(self):
        pass


class OldPolicy:
    def __init__(self, units, activations, lr, num_vars=60, max_actions=61, max_cons=60):
        self.units = units
        self.activations = activations
        self.num_vars = num_vars
        self.max_actions = max_actions
        self.max_cons = max_cons

        self.cons_embedding = Embedding(self.units, self.activations)
        # self.cons_embedding.build(input_shape=(1, num_vars+1))
        self.cuts_embedding = Embedding(self.units, self.activations)
        # self.cuts_embedding.build(input_shape=(1, num_vars+1))

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

            action_onehot = tf.cast(tf.one_hot(actions, len(prob)), tf.double)
            prob_selected = tf.reduce_sum(prob * action_onehot, axis=-1)

            prob_selected += 1e-8

            loss = -tf.reduce_mean(Qs * tf.math.log(prob_selected))
            weights = self.cons_embedding.weights_ + self.cons_embedding.biases_
            weights += self.cuts_embedding.weights_ + self.cuts_embedding.biases_
            gradients = tape.gradient(loss, weights)
            self.optimizer.apply_gradients(zip(gradients, weights))

        return loss.numpy()


class Policy:
    def __init__(self, units, activations):
        self.units = units
        self.activations = activations

        self.cons_embedding = Embedding(self.units, self.activations)
        self.cuts_embedding = Embedding(self.units, self.activations)

    def compute_prob(self, states):
        cons = np.concatenate((states[0], states[1].reshape(-1, 1)), axis=1).astype(float)
        cuts = np.concatenate((states[3], states[4].reshape(-1, 1)), axis=1).astype(float)

        h = self.cons_embedding(cons)
        g = self.cuts_embedding(cuts)

        S = tf.reduce_mean(tf.matmul(g, tf.transpose(h)), axis=1)
        return tf.cast(tf.nn.softmax(S), tf.double).numpy()

    def act(self, states):
        prob = self.compute_prob(states)
        prob /= np.sum(prob)
        action = np.asscalar(np.random.choice(len(prob), p=prob.flatten(), size=1))
        return action

    def get_weights(self):
        return self.cons_embedding.get_weights().copy(), self.cuts_embedding.get_weights().copy()

    def set_weights(self, cons_weights, cuts_weights):
        self.cons_embedding.set_weights(cons_weights)
        self.cuts_embedding.set_weights(cuts_weights)
