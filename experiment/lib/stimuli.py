import numpy as np
import itertools as it
from scipy.io import savemat
import os
import json
from collections import defaultdict
from toolz import *
from distributions import *

# ---------- Constructing environments ---------- #
DIRECTIONS = ('up', 'right', 'down', 'left')
ACTIONS = dict(zip(DIRECTIONS, it.count()))


BRANCH_DIRS = {
    2: {'up': ('right', 'left'),
        'right': ('up', 'down'),
        'down': ('right', 'left'),
        'left': ('up', 'down'),
        'all': ('right', 'left')},
    3: {'up': ('up', 'right', 'left'),
        'right': ('up', 'right', 'down'),
        'down': ('right', 'down', 'left'),
        'left': ('up', 'down', 'left'),
        'all': DIRECTIONS}
}

def move_xy(x, y, direction, dist=1):
    return {
        'right': (x+dist, y),
        'left': (x-dist, y),
        'down': (x, y+dist),
        'up': (x, y-dist),
    }.get(direction)


def dist(branch, depth):
    """Distance between nodes at a given depth of a tree with given branching factor."""
    if branch == 3:
        return 2 ** (depth - 1)
    else:
        return 2 ** (depth/2 - 0.5)
    
class Layouts:

    def cross(depth, mu, sigma, n=4, **kwargs):
        graph = {}
        layout = {}
        names = it.count()

        def direct(prev):
            if prev == 'all':
                yield from  ('right', 'down', 'left')
            else:
                yield prev
        
        def node(d, x, y, prev_dir):
            r = 0  # reward is 0 for now
            name = str(next(names))
            layout[name] = (x, y)
            graph[name] = {}
            if d > 0:
                for direction in direct(prev_dir):
                    x1, y1 = move_xy(x, y, direction, 1)
                    graph[name][direction] = (r, node(d-1, x1, y1, direction))                     
            return name

        def r_dist():
            from scipy.stats import norm
            d = norm(mu, sigma)
            vals = np.array([-3, -1, 1, 3]) * sigma + mu
            probs = [.1, .4, .4, .1]
            # vals = np.linspace(-1.5*sigma, 1.5*sigma, n)
            # delta = vals[1] - vals[0]
            # bins = np.array((-np.inf, *(vals[1:] - delta/2), np.inf))
            # probs = np.diff(d.cdf(bins))
            return Categorical(vals, probs)

        node(depth, 0, 0, 'all')
        rewards = r_dist().sample(len(graph)).tolist()
        return graph, layout, rewards

    def tree(branch, depth, mu, sigma, first='up', **kwargs):
        graph = {}
        layout = {}
        names = it.count()

        vals = np.array([-3, -1, 1, 3]) * sigma + mu
        R = Categorical(vals, [.1, .4, .4, .1])
        rewards = []

        def node(d, x, y, prev_dir):
            r = int(R.sample()) if rewards else 0 # received at this state
            name = str(next(names))
            rewards.append(r)
            layout[name] = (x, y)
            graph[name] = {}
            if d > 0:
                for direction in BRANCH_DIRS[branch][prev_dir]:
                    x1, y1 = move_xy(x, y, direction, dist(branch, d))
                    s1, r1 = node(d-1, x1, y1, direction)
                    graph[name][direction] = (r1, s1)
                                            
            return name, r


        node(depth, 0, 0, first)
        assert len(rewards) == len(graph), (len(rewards), len(graph))
        return graph, layout, rewards

    def _tree(branching, **kwargs):
        graph = {}
        layout = {}
        names = it.count()

        def node(d, x, y, prev_dir):
            r = 0  # reward is 0 for now
            name = str(next(names))
            layout[name] = (x, y)
            graph[name] = {}
            if d > 0:
                for direction in BRANCH_DIRS[branch][prev_dir]:
                    x1, y1 = move_xy(x, y, direction, dist(branch, d))
                    graph[name][direction] = (r, node(d-1, x1, y1, direction))
                                            
            return name

        node(depth, 0, 0, first)
        return graph, layout

    def heart(size):
        last_full = (size + 1)* 2

        @memoize
        def reward(s):
            return np.random.randint(-9, 8)


        def layer(h):            

            def full():
                x_max = 0.5 * h
                x = -x_max
                while x < x_max:
                    yield (x, h)
                    x += 1
            
            skip = h - last_full
            keep = size + 1 - skip
            if skip <= 0:
                yield from full()
            else:
                lay = drop(skip, full())
                yield from take(keep, lay)
                lay = drop(skip, lay)
                yield from take(keep, lay)

        layout = list(concat(layer(h) for h in range(1, 3 * size + 3)))
        r_layout = {n: i for i, n in enumerate(layout)}

        graph = [{} for n in layout]
        for n, (x, y) in enumerate(layout):
            left = r_layout.get((x - 0.5, y + 1))
            if left:
                graph[n]['left'] = (reward(left), left)
            right = r_layout.get((x + 0.5, y + 1))
            if right:
                graph[n]['right'] = (reward(right), right)
        labels = [reward(i) for i in range(len(graph))]
        return graph, layout, labels



def build(kind, **kwargs):
    return getattr(Layouts, kind)(**kwargs)


# ---------- Information about environments ---------- #

def n_step(graph):
    """Number of steps it takes to reach every state."""
    result = {}
    def search(s, n):
        result[s] = n
        for _, s1 in graph[s].values():
            search(s1, n+1)
    search('0', 0)
    return result

def get_paths(graph):
    """The sequences of actions and states that lead to each state."""

    state_paths = {}
    action_paths = {}
    def search(s, spath, apath):
        state_paths[s] = spath
        action_paths[s] = apath
        for a, (_, s1) in graph[s].items():
            search(s1, spath + [s1], apath + [a])
    search('0', ['0'], [])
    return {'state_paths': state_paths, 'action_paths': action_paths}


def transition_matrix(graph):
    """X[s0, s1] is 1 if there is an edge from s0 to s1, else 0."""
    X = np.zeros((len(graph), len(graph)))
    for s0, actions in graph.items():
        for _, s1 in actions.values():
            X[int(s0), int(s1)] = 1
    return X

def terminal(graph):
    x = np.zeros(len(graph))
    for s, actions in graph.items():
        if not actions:
            x[int(s)] = 1
    return x

def available_actions(graph):
    X = np.zeros((len(graph), len(ACTIONS)))
    for s0, actions in graph.items():
        for a in actions:
            X[int(s0), ACTIONS[a]] = 1
    return X


def trees():
    paths = defaultdict(dict)
    for branch in (2, 3):
        for depth in range(1, 6):
            graph, layout = build('tree', branch, depth)
            name = 'b{}d{}'.format(branch, depth)

            mat_dict = {
                'transition': transition_matrix(graph),
                'initial': 0,
                'terminal': terminal(graph),
                'actions': available_actions(graph),
                'branch': branch,
                'depth': depth,
            }
            savemat('env_data/{}.mat'.format(name), mdict=mat_dict)
            paths[branch][depth] = get_paths(graph)
    
    os.makedirs('env_data', exist_ok=True)
    with open('experiment/static/json/paths.json', 'w+') as f:
        json.dump(paths, f)

def main():
    from pprint import pprint
    pprint(build_cross(2))

if __name__ == '__main__':
    main()
