#!/usr/bin/env python3
import numpy as np
import random
from datetime import datetime
import sys

sys.path.append('lib')
from utils import dict_product, Labeler
from stimulator import Stimulator, stimulus
from stimuli import Layouts
from distributions import *

from envs import DeterministicGraphEnv
from agents import SearchAgent
# WARNING
# For some reason, we do not get consistent results despite the seed.
seed = 1
np.random.seed(seed)
random.seed(seed)


def subdict(d, keys):
    return {k: d[k] for k in keys}

class Stims(Stimulator):
    """Defines conditions and creates stimuli."""
    

    # ---------- Experiment structure ---------- #

    def conditions(self):
        conds = dict_product({
            'creation_date': str(datetime.now()),
            'clickDelay': 0,
            'moveDelay': 500,
            'encourage_planning': [False, True],
            # 'timeLimit': 240,
            # 'energy': 200,
            # 'moveEnergy': 2,
            # 'clickEnergy': 1,
            'depth': 3,
            'breadth': 2,
            'inspectCost': 1,
            'bonus_rate': 0.001
        })
        for c in conds:
            if c['encourage_planning']:
                c['mu'] = -4
                c['sigma'] = 16
            else:
                c['mu'] = 8
                c['sigma'] = 3
            yield c

    def blocks(self, params):
        # yield {'block': 'train_click', 'n_trial': 5}
        # yield {'block': 'train_energy', 'n_trial': 4}
        yield {'block': 'train_basic', 'n_trial': 5}
        yield {'block': 'train_hidden', 'n_trial': 5}
        yield {'block': 'train_ghost', 'n_trial': 5}
        yield {'block': 'train_final', 'n_trial': 5}
        yield {'block': 'test', 'n_trial': 20}

    def trials(self, params):
        for _ in range(params['n_trial']):
            yield from dict_product(params)

    # ---------- Create stimuli ---------- #

    def trial(self, params):
        graph, layout, rewards = Layouts.tree(params['breadth'], params['depth'], params['mu'], params['sigma'])

        return {
            'graph': graph,
            'layout': rescale(layout),
            # 'stateLabels': dict(zip(graph.keys(), graph.keys())),
            # 'stateRewards': dict(zip(graph.keys(), map(int, graph.keys()))),
            # 'stateLabels': list(range(len(graph))),
            'stateRewards': rewards,
            'stateLabels': 'reward',
            # 'stateDisplay': 'click',
            'edgeDisplay': 'never',
            # 'moveDelay': moveDelay,
            # 'clickDelay': clickDelay,
            'initial': 0
        }


def rescale(layout):
    # names, xy = zip(*enumerate(layout))
    names, xy = zip(*layout.items())
    x, y = np.array(list(xy)).astype('float').T
    # y *= -1
    x -= x.min()
    y -= y.min()
    # y *= 0.5
    # x *= 1.5
    xy = zip(x.tolist(), y.tolist())
    return dict(zip(names, xy))


def iter_rewards(rewards):
    for s0, choices in rewards.items():
        for st, r in choices.items():
            yield r

def map_rewards(rewards, func):
    return {s0: {st: func(r) for st, r in choices.items()}
            for s0, choices in rewards.items()}

def replace_state_names(rewards, names):
    return {names[s0]: {names[st]: r for st, r in choices.items()}
            for s0, choices in rewards.items()}


if __name__ == '__main__':
    s = Stims().run()
