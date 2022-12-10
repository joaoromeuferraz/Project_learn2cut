import numpy as np


def pad_state(s, num_rows=61):
    if s.shape[0] == num_rows:
        return s
    return np.pad(s, [(0, s.shape[0]-num_rows), (0, 0)])
