#!/usr/bin/env python3
from joblib import Parallel, delayed, load, dump
from tqdm import tqdm
import os
from agents import run_episode
from policies import LiederPolicy

from mouselab_utils import make_envs, ENV_TYPES
from utils import *
from contexttimer import Timer
from tqdm import tqdm

COST = 1.0

def encode_state(state):
    return ' '.join('_' if hasattr(x, 'sample') else str(int(x))
                    for x in state)

def run_rollouts(env_type, seed, n_env=1000, n_per_env=30, overwrite=False, i=0):
    file = f'data/traces/{env_type}_{seed}.pkl'
    if os.path.isfile(file) and not overwrite:
        print(f'{file} already exists. Skipping.')
        return

    try:
        polfile = f'data/policies/{env_type}_{seed}.pkl'
        pol = load(polfile)
    except FileNotFoundError:
        print(f'{polfile} not found. Skipping.')
        return

    envs = make_envs(COST, n_env, seed + 999, env_type)
    traces = []
    for env in tqdm(envs, desc=f'{env_type} {seed}', position=i):
        e_traces = []
        traces.append(e_traces)
        for i in range(n_per_env):
            e_traces.append(run_episode(pol, env))
    with Timer() as t:
        dump(traces, file)
    print(f'pickling time: {t.elapsed:.2f}')


os.makedirs('data/traces', exist_ok=True)
run_rollouts(1, 'constant_high', 1)

params = dict_product({'env_type': ENV_TYPES, 'seed': list(range(1,5))})
jobs = [delayed(run_rollouts)(**prm, i=i)
        for i, prm in enumerate(params)]
results = Parallel(n_jobs=12)(jobs)