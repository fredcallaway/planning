from mouselab import MouselabEnv
from distributions import Categorical
import numpy as np

from exact import solve
from analysis_utils import *

def make_env(mu, sigma, cost=1.00, scaling_factors=[1,1,1], branching=[3,1,2], seed=None, **kwargs):
    if seed is not None:
        np.random.seed(seed)
    
    def reward(depth):
        if depth > 0:
            x = np.array([-2,-1,1,2])
            vals = mu + sigma * x * scaling_factors[depth-1]
            return Categorical(vals).apply(round)
        return 0.

    return MouselabEnv.new_symmetric(branching, reward, cost=cost, **kwargs)





