import numpy as np
import tensorflow as tf
from layers import Embedding, AttentionEmbedding
from utils import pad_state, discounted_rewards
import tensorflow_probability as tfp


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


class Policy:
    def __init__(self, units, activations, lr=None, num_vars=60):
        self.units = units
        self.activations = activations
        self.attention = AttentionEmbedding(self.units, self.activations, num_vars)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        

    def compute_prob(self, states):
        S = self.attention(states)
        return tf.cast(tf.nn.softmax(S), tf.double)

    def act(self, states):
        prob = self.compute_prob(states).numpy()
        prob /= np.sum(prob)
        action = np.asscalar(np.random.choice(len(prob), p=prob.flatten(), size=1))
        return action

    def get_weights(self):
        return self.attention.trainable_variables.copy()

    def set_weights(self, weights):
        self.attention.set_weights(weights)
    
    def train(self, states, rewards, actions):
        losses = []
        gs =[]
        
        for state, reward, action in zip(states, rewards, actions):
            with tf.GradientTape() as tape:
                prob = tf.cast(tf.nn.softmax(self.attention(state), axis=-1), tf.double)
                action_onehot = tf.cast(tf.one_hot(action, len(prob)), tf.double)
                prob_selected = tf.reduce_sum(prob * action_onehot, axis=-1)
                prob_selected += 1e-8
                
                loss = -tf.reduce_mean(reward * tf.math.log(prob_selected))
                gradients = tape.gradient(loss, self.attention.trainable_variables)
                gs.append(gradients)
                self.optimizer.apply_gradients(zip(gradients, self.attention.trainable_variables))
                losses.append(loss.numpy())
        return losses, gs
                
class RandomPolicy:
    def __init__(self):
        pass
    def compute_prob(self, s):
        return None
    def get_weights(self):
        return [np.random.randn(10, 10)]
    def set_weights(self, x=None):
        pass
    def act(self, states):
        num_actions = states[3].shape[0]
        action = np.random.choice(num_actions)
        return action