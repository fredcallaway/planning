from itertools import combinations
from toolz import curry, concat
from functools import wraps

from distributions import ZERO
from models.utils import make_env
from models import MouselabModel


def observed(state, node):
    return not hasattr(state[node], 'sample')

env = make_env(0, 0)

def powerset(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return concat(combinations(s, r) for r in range(len(s)+1))

def path_values(env):
    vals = set()
    node_dists = [env.init[n] for n in env.path_to(env.leaves()[0])[1:]]
    for subset in powerset(node_dists):
        vals.update(map(float, sum(subset, ZERO).vals))
    # for node in env.path_to(env.leaves()[0])[1:]:
    #     total += env.init[node]
    #     vals.extend(total.vals)
    return sorted(vals)


class ClassicalModel(MouselabModel):
    """Model of a classical search algorithm plus satisficing and pruning."""
    def __init__(self, satisfice, prune):
        self.satisfice = satisfice
        self.prune = prune

    @staticmethod
    def param_choices(env):
        vals = path_values(env)
        return {
            'satisfice': [v for v in vals if v > 0],
            'prune': [v for v in vals if v < 0],
        }

    def preference(self, state, action):
        if action == env.term_action:
            if self.satisfice is not None:
                satisfied = env.expected_term_reward(state) >= self.satisfice
                return 1e10 if satisfied else -1e10
        elif self.prune is not None:
            if env.node_quality(action, state).expectation() <= self.prune:
                return -1e20
        return self.priority(state, action)

    def priority(self, env, state, node):
        """Returns the relative value of examining this node (vs another node)."""
        raise NotImplementedError()


class DepthFirstModel(ClassicalModel):
    """TODO"""
    def priority(self, state, action):
        # Don't click nodes not on the frontier.
        previous_nodes = env.path_to(action)[:-1]
        if not all(observed(state, node) for node in previous_nodes):
            return -1e10
        # Prefer nodes that are further from the start
        return len(previous_nodes)


class BreadthFirstModel(ClassicalModel):
    """TODO"""
    def priority(self, state, action):
        # Don't click nodes not on the frontier.
        previous_nodes = env.path_to(action)[:-1]
        if not all(observed(state, node) for node in previous_nodes):
            return -1e10
        # Prefer nodes that are closer to the start
        return -len(previous_nodes)


class BestFirstModel(ClassicalModel):
    """TODO"""
    def priority(self, state, action):
        previous_nodes = env.path_to(action)[:-1]
        if not all(observed(state, node) for node in previous_nodes):
            return -1e10
        q = env.node_quality(action, state)  # note backwards arguments!  plz don't ask why...
        return q.expectation()  # node_quality is a distribution, we just want the mean


class BackwardBestFirstModel(ClassicalModel):
    """TODO"""
    def priority(self, state, action):
        children = env.tree[action]
        if children and not any(observed(state, node) for node in children):
            return -1e10
        q = env.node_quality(action, state)
        return q.expectation()


class ProgressiveDeepeningModel(ClassicalModel):
    """TODO"""
    def priority(self, state, action,  last_click=None):
        previous_nodes = env.path_to(action)[:-1]
        if not all(observed(state, node) for node in previous_nodes):
            return -1e10
        if last_click:
            # version if last_click is available
            if last_click in env.leaves():
                if len(previous_nodes) == 1:
                    # the start of a new path = another leaf on the previous path
                    return len(env.path_to(last_click)[:-1])
                return len(previous_nodes)
            if last_click == previous_nodes[-1]:
                return 1e10
            return len(previous_nodes)
        else:
            # the following only works for a 3-1-2 ENVironment
            if action in env.leaves():
                leaf_group = None
                for subtree in env.tree:
                    if (len(subtree) == 2) and (action in subtree):
                        leaf_group = subtree
                for leaf in leaf_group:
                    if observed(state, leaf):
                        return 1 # set same value as the beginning of a new path
            return len(previous_nodes)

