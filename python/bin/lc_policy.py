#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from agents import run_episode
from mouselab_policy import MouselabPolicy
from mouselab_utils import make_env, ENV_TYPES, encode_state
from skopt import gp_minimize
from joblib import Parallel, delayed, dump, load
from toolz.curried import partition_all, concat, get
from tqdm import tqdm

def lc_policy(x, normalize_voi=True):
    x = np.array(x, dtype=float)
    if len(x) != 5:
        raise ValueError(f'len(x) == {len(x)} != 5')
    voi = x[2:]
    if normalize_voi and voi.sum():
        voi /= voi.sum()
    # Note: an `is_term` feature is functionally equivalent to a `cost` feature
    # when all clicks have the same cost.
    weights = dict(zip(['is_term', 'term_reward', 'voi_myopic', 'vpi_action', 'vpi_full'], x))
    return MouselabPolicy(weights)


BOUNDS = [
    (-100., 100.),  # is_term
    (-1, 1.),       # term_reward
    (0., 1.),       # voi_myopic
    (0., 1.),       # vpi_action
    (0., 1.),       # vpi_full
]

TRIALS = pd.read_json('../experiment/experiment/static/json/constant_high.json'
                      ).set_index('trial_id').stateRewards

def empirical_clicks(version):
    import sys
    sys.path.append('../experiment/lib')
    from analysis_utils import get_data

    data = get_data(version, '../experiment/data')
    tdf = data['mouselab-mdp'].query('block == "test"')
    def extract(q):
        return list(map(int, q['click']['state']['target']))
    
    return zip(tdf.trial_id, tdf.queries.apply(extract))

CLICKS = load('data/empirical_clicks.pkl')
CHUNKS = list(partition_all(5, CLICKS))
COST = 1.0
N_CALL = 50
N_JOB = 24


def yoked_rollout(pol, trial_id, clicks, n=1):
    env = make_env('constant_high', cost=COST, ground_truth=TRIALS[trial_id])
    true_init = env.init

    for _ in range(n):
        env.init = true_init
        trace = run_episode(pol, env)
        yield {'s': env.init, 'a': trace['actions'][0], 'q': sum(trace['rewards'])}

        for click in clicks:
            s = env.reset()
            env.init, r, *_ = env.step(click)
            q = r + sum(run_episode(pol, env)['rewards'])
            yield {'s': s, 'a': click, 'q': q}


# ---------- Policy Optimization ---------- #

def chunk_sum_returns(x, chunk_i):
    pol = lc_policy(x)
    chunk = CHUNKS[chunk_i]
    steps = concat(yoked_rollout(pol, *args) for args in chunk)
    return sum(map(get('q'), steps))

def write_policy(env_type, seed):
    name = f'{env_type}_{seed}'
    np.random.seed(seed)
    assert env_type == 'constant_high'
    best = -np.inf
    
    with Parallel(N_JOB) as parallel, tqdm(total=N_CALL, desc=name) as pbar:
        def loss(x):
            util = sum(parallel(delayed(chunk_sum_returns)(x, i) 
                                for i in range(len(CHUNKS))))
            pbar.update()
            nonlocal best
            best = max(best, util)
            pbar.set_postfix(last=util, best=best)
            return - util
    
        result = gp_minimize(loss, BOUNDS, n_calls=50, random_state=seed)

    result.specs['args'].pop('func')  # can't pickle
    pol = lc_policy(result.x)
    dump(result, f'data/gp_results/{name}.pkl')
    dump(pol, f'data/policies/{name}.pkl')


# ---------- Rollouts ---------- #

def chunk_write_rollouts(env_type, seed, chunk_i):
    polfile = f'data/policies/{env_type}_{seed}.pkl'
    pol = load(polfile)
    env = make_env('constant_high', cost=COST)
    chunk = CHUNKS[chunk_i]
    n = 0

    for i, (trial_id, clicks) in enumerate(chunk):
        data = {'s': [], 'a': [], 'q': [], 'phi': [], 'trial_id': trial_id}
        for step in yoked_rollout(pol, trial_id, clicks, n=300):
            data['s'].append(encode_state(step['s']))
            data['a'].append(step['a'])
            data['q'].append(step['q'])
            phi = pol.phi(step['s'], step['a'], compute_all=True)[1:]
            phi[0] = env.expected_term_reward(step['s'])
            data['phi'].append(phi)
    
        data['phi'] = np.stack(data['phi'])
        data['a'] = np.array(data['a'])
        data['q'] = np.array(data['q'])
        dump(data, f'data/rollouts/{env_type}_{seed}/{chunk_i}_{i}.pkl')
        n += len(data['q'])

    return n

def write_rollouts(env_type, seed):
    name = f'{env_type}_{seed}'
    os.makedirs(f'data/rollouts/{name}', exist_ok=True)
    np.random.seed(seed)
    assert env_type == 'constant_high'

    jobs = [delayed(chunk_write_rollouts)(env_type, seed, chunk_i)
            for chunk_i in range(len(CHUNKS))]
    n = sum(Parallel(N_JOB)(tqdm(jobs, desc=name)))
    print(f'{n} rollouts')


def main():
    os.makedirs('data/policies', exist_ok=True)
    os.makedirs('data/gp_results', exist_ok=True)
    os.makedirs('data/rollouts', exist_ok=True)
    # write_policy(ENV_TYPES[0], 1)
    write_rollouts(ENV_TYPES[0], 1)


    # jobs = [delayed(write_policy)(et, seed)
    #                 for et in ENV_TYPES
    #                 for seed in range(1,5)]
    # results = Parallel(n_jobs=16)(jobs)

if __name__ == '__main__':
    main()
