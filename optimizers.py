import numpy as np


class Optimizer(object):
    def __init__(self, w_policy):
        self.w_policy = w_policy.flatten()
