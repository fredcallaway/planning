#!/usr/bin/env python3

import numpy as np
from agents import Agent
from mouselab_policy import MouselabPolicy
from mouselab import MouselabEnv
from mouselab_utils import make_envs, get_util, ENV_TYPES
from skopt import gp_minimize
from joblib import Parallel, delayed, dump
from contexttimer import Timer

def lc_policy(x, normalize_voi=True):
    x = np.array(x)
    assert len(x) == 4, x
    voi = x[1:]
    if normalize_voi and voi.sum():
        voi /= voi.sum()
    # Note: an `is_term` feature is functionally equivalent to a `cost` feature
    # when all clicks have the same cost.
    weights = dict(zip(['is_term', 'voi_myopic', 'vpi_action', 'vpi_full'], x))
    return MouselabPolicy(weights)


def avg_utility(x,train_envs):
    with Timer() as t:
        util = get_util(lc_policy(x), train_envs)
        xs =  np.array2string(np.array(x), 
                              formatter={'float_kind': lambda x: f'{x: 6.2f}'})
        print(f'{xs} -> {util:6.3f}  ({t.elapsed:2.1f} seconds)')
    return util

bounds = [
    (-30., 30.), # is_term
    (0., 1.),    # voi_myopic
    (0., 1.),    # vpi_action
    (0., 1.),    # vpi_full
]


cost = 1.00
def write_policy(env_type, seed):
    print(env_type, seed)
    envs = make_envs(cost, 100, seed, env_type)
    def loss(x):
        return - avg_utility(x,envs) 
    
    result = gp_minimize(loss, bounds, n_calls=50, random_state=seed)
    result.specs['args'].pop('func')  # can't pickle
    pol = lc_policy(result.x)
    dump(result, f'data/gp_results/{env_type}_{seed}.pkl')
    dump(pol, f'data/policies/{env_type}_{seed}.pkl')

jobs = [delayed(write_policy)(et, seed)
                for et in ENV_TYPES
                for seed in range(1,5)]
results = Parallel(n_jobs=16)(jobs)