from gymrats.policies import Policy
from bayespy.nodes import GaussianARD, GaussianGamma, SumMultiply, Gamma

from bayespy.inference import VB
import numpy as np

class BayesianRegression(object):
    """Bayesian linear regression."""
    def __init__(self, n_feature, prior_mean=0, prior_precision=1e-6, prior_a=1, prior_b=1):
        super().__init__()
        self.weights = GaussianARD(0, 1e-6, shape=(n_feature,))

        if np.shape(prior_mean) != (n_feature,):
            prior_mean = prior_mean * np.ones(n_feature)
        if np.shape(prior_precision) != (n_feature, n_feature):
            prior_precision = prior_precision * np.identity(n_feature)
        self.prior_mean = prior_mean
        self.prior_precision = prior_precision
        self.prior_a = prior_a
        self.prior_b = prior_b
        self.weights = GaussianGamma(self.prior_mean, self.prior_precision, self.prior_a, self.prior_b)
        # self.print()


    def fit(self, X, y):
        self.weights = GaussianGamma(self.prior_mean, self.prior_precision, self.prior_a, self.prior_b)
        F = SumMultiply('i,i', self.weights, X)
        y_obs = GaussianARD(F, 1)
        y_obs.observe(y)

        Q = VB(y_obs, self.weights)
        Q.update(repeat=1, verbose=False)  # exact inference, no need to iterate

    def predict(self, x, return_var=False):
        y = SumMultiply('i,i', self.weights, x)
        y_hat, var, *_ = y.get_moments()
        if return_var:
            return y_hat, var
        else:
            return y_hat

    def print(self):
        q, var = self.weights.get_moments()[:2]
        print(f'{q}\n{var}')


from tqdm import tqdm

class BayesianQLearner(Policy):
    """Learns a linear Q function by Bayesian regression."""
    def __init__(self, n_feature, **kwargs):
        super().__init__(**kwargs)
        assert n_feature == 5
        self.Q = BayesianRegression(5, 0.,
            np.r_[np.zeros(4), 1],
            1e-3 * np.ones(5)
            # np.r_[np.ones(4) * 1e-1, 1e3]
            )
        self.data = {
            'phis': [],
            'qs': [],
        }

    def attach(self, agent):
        super().attach(agent)

    def finish_episode(self, trace):
        if (self.i_episode % 10) == 0:
            self.Q.print()

        returns = np.flip(np.cumsum(np.flip(trace['rewards'], 0)), 0)
        self.data['qs'].extend(returns)
        assert len(self.data['qs']) == len(self.data['phis'])
        self.Q.fit(np.stack(self.data['phis']), np.array(self.data['qs']))

    def act(self, state):
        actions = list(self.env.actions(state))
        phi = list(map(self.env.action_features, actions))

        q, var = self.Q.predict(phi, return_var=True)
        i = np.argmax(q + np.sqrt(var) * np.random.randn())
        self.data['phis'].append(phi[i])
        return actions[i]


