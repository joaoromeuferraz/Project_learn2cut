import numpy as np


def pad_state(s, num_rows=61):
    if s.shape[0] == num_rows:
        return s
    return np.pad(s, [(0, s.shape[0]-num_rows), (0, 0)])

def discounted_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_sum = 0
    for i in reversed(range(0,len(r))):
        discounted_r[i] = running_sum * gamma + r[i]
        running_sum = discounted_r[i]
    return list(discounted_r)


class AdamOptimizer:
    def __init__(self, lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m_dw, self.v_dw = 0, 0
        self.t = 1
        
    def update(self, w, dw):
        # momentum
        self.m_dw = self.beta_1*self.m_dw + (1-self.beta_1)*dw
        
        #rms
        self.v_dw = self.beta_2*self.v_dw + (1-self.beta_2)*(dw**2)
        
        # bias correction
        m_dw_corr = self.m_dw/(1-self.beta_1**self.t)
        v_dw_corr = self.v_dw/(1-self.beta_2**self.t)
        
        # update weights
        w = w - self.lr*(m_dw_corr/(np.sqrt(v_dw_corr) + self.epsilon))
        
        self.t += 1
        
        return w
   