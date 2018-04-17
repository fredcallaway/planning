from agents import TrueModel
from abc import ABC, abstractmethod

from utils import value_iteration

# ================================== #
# ========= Pseudo Rewards ========= #
# ================================== #


class PseudoRewarder(ABC):
    """Doles out pseudo-rewards (PRs)."""
    def __init__(self, env, freq=1, model=None, V=None):
        model = model or TrueModel(env)
        V = V or value_iteration(env, discount=1)

        self.env = env
        self.model = model
        self.V = V
        self.freq = freq

    def recover(self, s0, a, s1):
        return self._cache.get(s1, 0)

    def start_episode(self, s0):
        self._cache = self._get_cache(s0)

    def update(self, s1):
        if s1 in self._cache:
            self._cache = self._get_cache(s1)

    @abstractmethod
    def _get_cache(self, s0):
        return

    def _horizon(self, s0):
        for path in self.model.paths(s0, depth=self.freq):
            yield path[-1].s1

class ObservablePseudo(PseudoRewarder):
    """All PRs are visible all the time."""

    def _get_cache(self, s0):
        assert 0  # this makes too many assumptions
        cache = {}
        q = [s0]
        while q:
            s0 = q.pop()
            for s1 in self._horizon(s0):
                assert s1 not in cache
                cache[s1] = self.V[s1] - self.V[s0]
                q.append(s1)
            

class HorizonPseudo(PseudoRewarder):
    """Only shows PRs n states away from the last PR (or initial state)."""
    def __init__(self, env, f=None, **kwargs):
        super().__init__(env, **kwargs)
        self.f = f or (lambda pr, s0, s1: pr.V[s1])
        self.rewards = self._compute_caches()

    def _compute_caches(self):
        if self.freq == 0:
            return {}
        return {s0: {s1: self.f(self, s0, s1) for s1 in self._horizon(s0)
                     if not self.model.is_final(s1)}
                for s0 in range(self.model.n_states)}

    def _get_cache(self, s0):
        return self.rewards.get(s0, {})


class OnlinePseudo(HorizonPseudo):
    """Computes pseudo-rewards online."""

    def _horizon(self, s0):
        for path in self.model.paths(s0, depth=self.freq):
            yield path[-1].s1

    def _compute_horizon(self, s0):
        # TODO: handle different length paths
        # TODO: handle multiple paths to one state
        for s1 in set(self._horizon(s0)):
            assert s1 not in self._cache
            self._cache[s1] = self.V[s1] - self.V[s0]


class PrecomputedPseudoRewarder(HorizonPseudo):
    """Doles out precomputed pseudo-rewards."""
    def __init__(self, state_caches):
        self.rewards = state_caches

    @classmethod
    def from_json(cls, file):
        import json
        with open(file) as f:
            return cls(json.load(f))



