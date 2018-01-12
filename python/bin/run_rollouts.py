#!/usr/bin/env python3
from joblib import Parallel, delayed, load, dump
from tqdm import tqdm
import os
from agents import Agent
from policies import LiederPolicy

from collections import defaultdict
from mouselab_utils import make_envs, ENV_TYPES, encode_state
from utils import *
from contexttimer import Timer
from tqdm import tqdm

COST = 1.0


def run_rollouts(env_type, seed, n_env=1000, n_per_env=30, overwrite=False, i=0):
    file = f'data/rollouts/{env_type}_{seed}.pkl'
    if os.path.isfile(file) and not overwrite:
        print(f'{file} already exists. Skipping.')
        return

    try:
        polfile = f'data/policies/{env_type}_{seed}.pkl'
        pol = load(polfile)
    except FileNotFoundError:
        print(f'{polfile} not found. Skipping.')
        return

    agent = Agent()
    agent.register(pol)
    envs = make_envs(COST, n_env, seed + 999, env_type)
    data = defaultdict(list)
    for env_i, env in enumerate(tqdm(envs, desc=f'{env_type} {seed}', position=i)):
        agent.register(env)
        for _ in range(n_per_env):
            trace = agent.run_episode()
            q = cum_returns(trace['rewards'])
            for s, a, q in zip(trace['states'], trace['actions'], q):
                data['env_i'].append(env_i)
                data['s'].append(encode_state(s))
                data['a'].append(a)
                data['q'].append(q)
                data['phi'].append(pol.phi(s, a)[:5])

    for k, v in data.items():
        if k != 's':
            data[k] = np.array(v)
    dump(data, file)


def main():
    # os.makedirs('data/rollouts', exist_ok=True)
    # run_rollouts('constant_high', -1, 20, 3)

    params = list(dict_product({'env_type': ENV_TYPES, 'seed': list(range(1,5))}))
    jobs = [delayed(run_rollouts)(**prm, i=i)
            for i, prm in enumerate(params)]
    Parallel(n_jobs=len(params))(jobs)

if __name__ == '__main__':
    main()
