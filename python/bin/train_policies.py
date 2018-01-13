#!/usr/bin/env python3
import os
import numpy as np
from agents import Agent
from mouselab_policy import MouselabPolicy
from mouselab import MouselabEnv
from mouselab_utils import make_envs, get_util, ENV_TYPES
from skopt import gp_minimize
from joblib import Parallel, delayed, dump
from contexttimer import Timer

parallel = Parallel(40)
COST = 1.0

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


def avg_utility(x, train_envs):
    with Timer() as t:
        util = get_util(lc_policy(x), train_envs, parallel)
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

def write_policy(env_type, seed):
    print(env_type, seed)
    envs = make_envs(COST, 800, seed, env_type)

    iters = iter(range(1000))
    
    def loss(x):
        with Timer() as t:
            util = get_util(lc_policy(x), envs, parallel)
            xs =  np.array2string(np.array(x), 
                                  formatter={'float_kind': lambda x: f'{x: 6.2f}'})
            print(f'{next(iters)}  {xs} -> {util:6.3f}  ({t.elapsed:2.1f} seconds)')
        return - util
    
    result = gp_minimize(loss, bounds, n_calls=50, random_state=seed)
    result.specs['args'].pop('func')  # can't pickle
    pol = lc_policy(result.x)
    dump(result, f'data/gp_results/{env_type}_{seed}.pkl')
    dump(pol, f'data/policies/{env_type}_{seed}.pkl')


def main():
    os.makedirs('data/policies', exist_ok=True)
    os.makedirs('data/gp_results', exist_ok=True)
    write_policy(ENV_TYPES[0], 1)


    # jobs = [delayed(write_policy)(et, seed)
    #                 for et in ENV_TYPES
    #                 for seed in range(1,5)]
    # results = Parallel(n_jobs=16)(jobs)

if __name__ == '__main__':
    main()
