from collections import namedtuple, defaultdict, deque, OrderedDict
import numpy as np
import sys
from io import StringIO
import json
import itertools as it
import json
import toolz.curried as tz

import gym
from gym.envs.toy_text.discrete import DiscreteEnv
from scipy.stats.distributions import bernoulli, norm

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


EMPTY = 0
WALL = 1
GOAL = 2


# One possible result of an action
# p: probability of this result
# s1: next state
# r: reward
# done: True if the episode is over else False
Result = namedtuple('Result', ['p', 's1', 'r', 'done'])



class DeterministicGraphEnv(DiscreteEnv):
    """An environment specified by a graph."""
    default_spec = {
        'initial': 'A',
        'final': ['C'],
        'graph': {
            'A': {
                'left': ('B', 5),
                'right': ('C', 0)
            },
            'B': {
                'left': ('C', 5),
                'right': ('A', 0)
            },
            'C': {
                'left': ('A', 5),
                'right': ('B', 0)
            },
        }
    }
    def __init__(self, spec=None):
        initial = spec['initial']
        graph = spec['graph']

        self.actions = sorted(set(tz.concat(c.keys() for c in graph.values())))
        n_states = len(graph)

        self.action_idx = {a: i for i, a in enumerate(self.actions)}
        n_actions = len(self.actions)
        
        initial_state = np.zeros(n_states)
        initial_state[initial] = 1

        # P[s][a] == [(probability, nextstate, reward, done), ...]
        # By default, an action has no effect.
        P = {s: {a: [(1, s, 0, False)] for a in range(n_actions)}
             for s in range(n_states)}

        for s, choices in graph.items():
            for a, (r, s1) in choices.items():
                done = not graph[s1]
                a = self.action_idx[a]
                P[int(s)][a] = [(1, int(s1), r, done)]

        super().__init__(n_states, n_actions, P, initial_state)

    @property
    def n_states(self):
        return len(self.state_idx)
        
    def to_json(self, file):
        with open(file, 'w+') as f:
            json.dump(self.spec, f)

    @classmethod
    def from_json(cls, file):
        with open(file) as f:
            spec = json.load(f, object_pairs_hook=OrderedDict)
            return DeterministicGraphEnv(spec)


    @classmethod
    def random_tree(cls, height, branch=2, reward=None):
        if reward is None:
            reward = lambda depth: np.random.uniform(-10, 10)
        
        q = deque()  # queue of states to expand
        graph = []   # list of (s0, [(s1, reward)])
        ids = it.count()
        final = set()
        def expand(s0, depth):
            if depth == height:
                final.add(s0)
                graph.append((s0, []))
                return
            options = []
            for s1 in range(s0 + 1, s0 + 1 + branch):
                s1 = next(ids)
                options.append((s1, reward(depth)))
                q.append((s1, depth + 1))
            graph.append((s0, options))

        q.append((next(ids), 0))
        while q:
            expand(*q.popleft())

        spec = {
            'initial': 0,
            'final': lambda s: s in final,
            'graph': graph
        }
        return DeterministicGraphEnv(spec)
        # return spec


class PachinkoEnv(DeterministicGraphEnv):
    """A graph environment that looks like pachinko.
     
      O   O   O   O   O   O   O
    O   O   O   O   O   O   O  
      O   O   O   O   O   O   O
    O   O   O   O   O   O   O  
    """

    reward_funcs = {
        'uniform': lambda d, w, c: np.random.randint(-5, 6),
        'mult': lambda d, w, c: np.random.randint(-5, 6) * d,
        'exp': lambda d, w, c: np.random.randint(-5, 6) ** d,
        'norm': lambda d, w, c: int(np.random.normal() * 10),
        'norm_depth': lambda d, w, c: int(np.random.normal() * d)
    }
    def __init__(self, width, length, initial='0_0', reward='uniform'):
        self.width = width
        self.length = length
        self.initial = initial
        self.reward = reward
      
        # Construct spec
        if not callable(reward):
            reward = self.reward_funcs[reward]

        def edge(d, w, c):
            w1 = w + c - (d % 2)
            s1 = '{}_{}'.format(d+1, w1)
            return (s1, reward(d, w, c))

        final =[]
        def node(d, w):
            s0 = '{}_{}'.format(d, w)
            if d == length - 1:
                final.append(s0)
                return (s0, {})
            elif d % 2 and w == 0:  # must go down
                return (s0, {'down': edge(d, w, 1)})
            elif not (d % 2) and w == width - 1:  # must go up
                return (s0, {'up': edge(d, w, 0)})
            else:  # two choices
                return (s0, {'up': edge(d, w, 0), 'down': edge(d, w, 1)})

        graph = [node(d, w) 
                 for d in range(length)
                 for w in range(width)]

        spec = {
            'initial': initial,
            'final': final,
            'graph': OrderedDict(graph)
        }
        super().__init__(spec)

def main():
    # env = DecisionTreeEnv.random(2)
    env = DeterministicGraphEnv.random_tree(2)
    print(env['graph'])
    # env.to_json('plane.json')


def test_pachinko():
    e1 = PachinkoEnv(2, 2)
    e1.to_json('test.json')
    e2 = PachinkoEnv.from_json('test.json')
    assert e1.P == e2.P



if __name__ == '__main__':
    test_pachinko()















