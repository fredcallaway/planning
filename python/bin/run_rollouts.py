#!/usr/bin/env python3
from joblib import Parallel, delayed, load, dump
from tqdm import tqdm
import os
from agents import Agent

from collections import defaultdict
from mouselab_utils import make_envs, ENV_TYPES, encode_state
from mouselab_policy import MouselabPolicy
from utils import *
from contexttimer import Timer

COST = 1.0

def run_rollouts(env_type, pol_seed, env_seed=None, n_env=1000, n_per_env=30, overwrite=False, i=0):
    if env_seed is None:
        env_seed = pol_seed + 999
    file = f'data/rollouts/{env_type}_{pol_seed}_{env_seed}.pkl'
    if os.path.isfile(file) and not overwrite:
        print(f'{file} already exists. Skipping.')
        return

    try:
        polfile = f'data/policies/{env_type}_{pol_seed}.pkl'
        pol = load(polfile)
        agent = Agent()
        agent.register(pol)
    except FileNotFoundError:
        print(f'{polfile} not found. Skipping.')
        return

    envs = make_envs(COST, n_env, env_seed, env_type)

    # Ensures that all features are computed.
    phi_pol = MouselabPolicy(dict(voi_myopic=1, vpi_action=1, vpi_full=1))
    agent.register(phi_pol)
    def phi(s, a):
        return phi_pol.phi(s, a)[:5]

    data = defaultdict(list)
    for env_i, env in enumerate(tqdm(envs, desc=f'{env_type} {pol_seed}', position=i)):
        agent.register(env)
        for _ in range(n_per_env):
            trace = agent.run_episode()
            q = cum_returns(trace['rewards'])
            for s, a, q in zip(trace['states'], trace['actions'], q):
                data['env_i'].append(env_i)
                data['s'].append(encode_state(s))
                data['a'].append(a)
                data['q'].append(q)
                data['phi'].append(phi(s, a))

    for k, v in data.items():
        if k != 's':
            data[k] = np.array(v)
    dump(dict(data), file)


def main():
    os.makedirs('data/rollouts', exist_ok=True)
    params = list(dict_product({
        'env_type': 'constant_high',
        'pol_seed': 1,
        'env_seed': list(range(10, 40)),
        'n_env': 100,
        'n_per_env': 60
    }))

    jobs = [delayed(run_rollouts)(**prm, i=i)
            for i, prm in enumerate(params)]
    Parallel(n_jobs=len(params))(jobs)

if __name__ == '__main__':
    main()
