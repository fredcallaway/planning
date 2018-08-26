
import numpy as np
from toolz import memoize

from gymrats.core import Agent
from gymrats.policies import SoftmaxPolicy

class ErrorModel():
    """Computes likelihoods for a softmax policy on a grid of temp and p_error."""
    def __init__(self, env, preference, data):
        self.env = env
        self.policy = SoftmaxPolicy(preference)
        self.data = data
        Agent(self.env, self.policy)  # register env with policy.
        self.prefs = np.stack(self.data.state.apply(self.policy.preferences))
        self.prefs -= self.prefs.max(1).reshape(-1, 1)  # prevent float overflow 

        idx = np.arange(len(data))
        self.chosen = (idx, data.action.as_matrix())

    def likelihood(self, temp=1e-9, p_error=None):
        """Returns likelihood for all combinations of temp and p_error given.
        
        Return value has shape (len(self.data), len(temp), len(p_error)).
        """
        temp = np.atleast_1d(temp)
        ep = np.exp(np.einsum('ij,k -> ijk', self.prefs, 1/temp))
        probs = ep[self.chosen] / ep.sum(1)  # shape: (state, temp)
        if p_error is not None:
            p_error = np.atleast_1d(p_error)
            probs = (probs[..., np.newaxis] * (1-p_error) + 
                     self.random_likelihood[..., np.newaxis] * p_error)
        return probs

    def maximum_likelihood_estimate(self, temp, p_error):
        logp = np.log(self.likelihood(temp, p_error)).sum(0)
        t, e = np.unravel_index(np.argmax(logp), logp.shape)
        return {'logp': logp[t, e], 'temp': temp[t], 'p_error': p_error[e]}

    @property
    @memoize
    def random_likelihood(self):
        no_pref = lambda *_: 0
        return ErrorModel(self.env, no_pref, self.data).likelihood()