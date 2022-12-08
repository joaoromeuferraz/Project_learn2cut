import numpy as np


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
