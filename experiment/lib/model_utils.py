import numpy as np
from agents import TrueModel

# ---------- Environments ---------- #

def env_index(env_file):
    return int(env_file.rsplit('_')[1].split('.json')[0])

def state_row(s_str):
    return int(s_str.split('_')[1])

# I have to do all sorts of awful mapping between
# different ways of referring to states.
def fix_str_path(path):
    if not isinstance(path[0], str):
        return path  # already fixed
    return np.array([int(s.split('_')[1])
                     for s in path])

def loc_to_state(env, step, height):
    return env.state_idx['{}_{}'.format(step, height)]

def all_paths(env, step, row, depth=-1):
    model = TrueModel(env)
    return np.array([[row]+[state_row(env.states[n.s1]) for n in path]
                     for path in model.paths(loc_to_state(env, step, row), depth=depth)])

def state_to_idx(s):
    return tuple(map(int, s.split('_')))