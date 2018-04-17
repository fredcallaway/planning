from collections import namedtuple, defaultdict, Counter, deque
import numpy as np
from abc import ABC, abstractmethod
import utils

np.set_printoptions(precision=3, linewidth=200)
MAX_STEPS = 1000
API_KEY = 'sk_R6mkDKdZTMC3deGMZi1Slg'



# ========================== #
# ========= Agents ========= #
# ========================== #

class Agent(ABC):
    """An agent that can run openai gym environments."""
    def __init__(self, env, discount=0.99):
        self.env = env
        self.i_episode = 0
        self.explored = set()
        self.discount = discount

        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n

    @abstractmethod
    def act(self, state):
        pass

    def update(self, state, action, new_state, reward, done):
        pass

    def start_episode(self, state):
        pass

    def run_episode(self, render=False, max_steps=1000, interact=False):

        if interact:
            render = 'human'
            last_cmd = ''
        trace = {
            'i_episode': self.i_episode,
            'states': [],
            'actions': [],
            'rewards': [],
            'finished': False,
            'return': None
        }
        new_state = self.env.reset()
        self.start_episode(new_state)
        done = False
        for i_step in range(max_steps):
            state = new_state
            self.explored.add(state)

            self.render(render)

            if interact:
                cmd = input('> ') or last_cmd
                if cmd == 'break':
                    import ipdb; ipdb.set_trace()
                if cmd == 'exit':
                    exit()

            action = self.act(state)
            new_state, reward, done, info = self.env.step(action)
            self.update(state, action, new_state, reward, done)
            
            trace['states'].append(state)
            trace['actions'].append(action)
            trace['rewards'].append(reward)

            if done:
                trace['finished'] = True
                self.render(render)
                break

        trace['states'].append(new_state)  # final state
        self.i_episode += 1
        trace['return'] = sum(trace['rewards'])   # TODO discounting
        return trace

    def render(self, mode):
        if mode == 'step':
            input('> ')
            utils.clear_screen()
            self.env.render()
        elif mode == 'clear':
            utils.clear_screen()
            self.env.render()
        elif mode:
            self.env.render(mode=mode)

    def run_many(self, n_episodes, **kwargs):
        data = {
            'i_episode': [],
            'n_steps': [],
            'return': [],
            'finished': [],
        }
        for _ in range(n_episodes):
            trace = self.run_episode(**kwargs)
            data['i_episode'].append(trace['i_episode'])
            data['n_steps'].append(len(trace['states']))
            data['return'].append(trace['return'])
            data['finished'].append(trace['finished'])

        return data


class PlanAgent(Agent):
    """An Agent with a plan."""
    def __init__(self, env, replan=False, **kwargs):
        super().__init__(env, **kwargs)
        self.replan = replan
        self.plan = iter([])

    def act(self, state):
        try:
            if self.replan:
                raise StopIteration()
            else:
                return next(self.plan)
        except StopIteration:
            self.plan = iter(self.make_plan(state))
            return next(self.plan)


class SearchAgent(PlanAgent):
    """Searches for the maximum reward path using a model."""

    def __init__(self, env, depth=None, model=None, memory=False, **kwargs):
        super().__init__(env, **kwargs)
        if depth is None:
            depth = -1  # infinite depth
        self.depth = depth

        if model is None:
            model = TrueModel(env)
        self.model = model
        self.memory = memory
        self.last_state = None

    def reward(self, s0, a, s1, r):
        return r

    def update(self, s0, a, s1, r, done):
        if self.memory:
            # TODO longer memory (should take into account recency of
            # a repeated state).
            self.last_state = s0

    def make_plan(self, state):
        def eval_path(path):
            # Choose path with greatest reward. In case of a tie, prefer the path
            # that takes you to unexplored territory. If no such path exists,
            # don't go back to the state you were at previously.
            reward = sum((self.reward(*node[1:])) * self.discount ** i
                         for i, node in enumerate(path))
            num_new = sum(node.s1 not in self.explored for node in path)
            not_backwards = all(node.s1 != self.last_state for node in path)
            return (reward, num_new, not_backwards)
        path = max(self.model.paths(state, depth=self.depth), key=eval_path)
        return (node.a for node in path)


class PseudoAgent(SearchAgent):
    """SearchAgent that makes decisions based partially on pseudo-rewards."""

    def __init__(self, env, pseudo_rewarder, pseudo_weight=1, **kwargs):
        super().__init__(env, **kwargs)
        if callable(pseudo_rewarder):
            pseudo_rewarder = pseudo_rewarder(env)
        self.pseudo_rewarder = pseudo_rewarder
        self.pseudo_weight = pseudo_weight

    def start_episode(self, state):
        self.pseudo_rewarder.start_episode(state)
        super().start_episode(state)

    def reward(self, s0, a, s1, r):
        pseudo = self.pseudo_rewarder.recover(s0, a, s1)
        return r + self.pseudo_weight * pseudo

    def update(self, s0, a, s1, r, done):
        if not done and self.pseudo_rewarder:
            self.pseudo_rewarder.update(s1)
        super().update(s0, a, s1, r, done)


class RandomAgent(Agent):
    """A not-too-bright Agent."""
    def __init__(self, env):
        super().__init__(env)
    
    def act(self, state):
        return self.env.action_space.sample()
      

    
# ========================== #
# ========= Models ========= #
# ========================== #

class Model(object):
    """Learned model of an MDP."""
    Node = namedtuple('Node', ['p','s0', 'a', 's1', 'r'])

    def __init__(self, env):
        # (s, a) -> [total_count, outcome_counts]
        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.counts = defaultdict(lambda: [0, Counter()])

    def results(self, s, a):
        """Yields possible outcomes of taking action a in state s.

        Outcome is (prob, s1, r, done).
        """
        total_count, outcomes = self.counts[s, a]
        for (s1, r, done), count in outcomes.items():
            yield (count / total_count, s1, r, done)

    def expected_reward(self, s0, a, s1=None):
        return sum(p * r for (p, s1_, r, done) in self.results(s0, a)
                   if s1 is None or s1 == s1_)

    def update(self, s0, a, s1, r, done):
        self.counts[s0, a][0] += 1  # total count
        self.counts[s0, a][1][s1, r, done] += 1  # outcome count

    def paths(self, state, depth=-1, target=None, cycles=False):
        """Yields all paths to a final state or a state `depth` steps away."""

        def expand(path, explored, depth):
            # TODO explored!
            s0 = path[-1].s1 if path else state  # state on initial call
            for a in range(self.n_actions):
                for (p, s1, r, done) in self.results(s0, a):
                    if s1 == s0:
                        continue
                    if p and (cycles or s1 not in explored):
                        new_path = path + [self.Node(p, s0, a, s1, r)]
                        complete = (s1 == target
                                    if target is not None
                                    else done or depth ==1)
                        if complete:                        
                                yield new_path
                        else: 
                            yield from expand(new_path, explored | {s1}, depth-1)

        yield from expand([], {state}, depth)

    def is_final(self, state):
        return list(self.paths(state)) == []


class TrueModel(Model):
    """Accurate model of a DiscreteEnv."""
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

    def results(self, state, action):
        yield from self.env.P[state][action]

    def update(*args):
        pass


class ModelBasedAgent(Agent):
    """Agent that builds a model of the MDP."""
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.model = Model(env)
        self.Q = np.zeros((self.model.n_states, self.model.n_actions))

    def act(self, state):
        noise = np.random.randn(self.Q.shape[1]) / (self.i_episode + 1)
        return np.argmax(self.Q[state] + noise)  # a = action

    def update(self, s0, a, s1, r, done):
        self.model.update(s0, a, s1, r, done)
        self.update_policy(s0, a)

    def update_policy(self, s, a):
        Q, V = self.Q, self.V
        expected_future_reward = sum(p * V(s1) for p, s1, r, done in self.model.results(s, a))
        Q[s, a] = self.model.expected_reward(s, a) + self.discount * expected_future_reward

    def V(self, s):
        """The value of taking the best possible action from state s."""
        return self.Q[s].max()


class PrioritizedSweeping(ModelBasedAgent):
    """Learns by replaying past experience.

    https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume4/kaelbling96a-html/node29.html
    """
    def __init__(self, env, n_simulate=0, **kwargs):
        super().__init__(env, **kwargs)
        self.n_simulate = n_simulate
        self.value_change = np.zeros(self.n_states)
        self.predecessors = defaultdict(set)

    def update(self, s0, a, s1, r, done):
        super().update(s0, a, s1, r, done)

        self.predecessors[s1].add(s0)
        # Update Q by simulation.
        for _ in range(self.n_simulate):
            s = self.value_change.argmax()
            if self.value_change[s] == 0:
                break  # no new information to propogate
            self.value_change[s] = 0  # reset
            self.update_policy(s, self.act(s))

    def update_policy(self, s0, a):
        # Track changes to prioritize simulations.
        old_val = self.V(s0)
        super().update_policy(s0, a)
        change = abs(self.V(s0) - old_val)
        for s_pred in self.predecessors[s0]:
            self.value_change[s_pred] = max(self.value_change[s_pred], change)  # TODO weight by transition prob







