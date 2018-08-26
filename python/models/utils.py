from mouselab import MouselabEnv
from distributions import Categorical
import numpy as np

from exact import solve
from analysis_utils import *

SCALING = {
    'increasing': [1/2, 1, 6],
    'decreasing': [6, 1, 1/2],
    'constant': [1, 1, 1]
}

def make_env(mu, sigma, cost=1.00, scaling_factors=[1,1,1], branching=[3,1,2], seed=None, **kwargs):
    """Returns an environment with structure similar to those used in the experiment."""
    if seed is not None:
        np.random.seed(seed)
    
    def reward(depth):
        if depth > 0:
            x = np.array([-2,-1,1,2])
            vals = mu + sigma * x * scaling_factors[depth-1]
            return Categorical(vals).apply(round)
        return 0.

    return MouselabEnv.new_symmetric(branching, reward, cost=cost, **kwargs)

# def make_exp_env(exp, **kwargs):
#     if exp == 1:
#         return make_env(0, 5, **kwargs)
#     elif exp == 2:
#         assert 0
#         return make_env(0, 4, **kwargs)
#     else:
#         raise ValueError('exp must be 1 or 2.')
        


def fetch_data(exp, version=None):
    """Returns a dict of cleaned DataFrames for the given version number.

        participants: one row for each participant
        trials: one row for each test trial
        unrolled: one row for each meta-action (clicking or terminating)

    Participants are excluded if either (1) they did not click any nodes
    during the block which explicitly asks them to click or (2) they answered
    incorrectly on more than one attention check question.
    """
    if exp not in (1, 2):
        raise ValueError('exp must be 1 or 2.')

    if version is None:
        version = 'c1.1' if exp is 1 else 'c2.1'

    exp_data = get_data(version, '../experiment/data')

    pdf = exp_data['participants'].set_index('pid')
    complete = pdf.completed
    pdf = pdf.loc[complete]
    if 'variance' in pdf:
        pdf.variance = pdf.variance.replace(2442, 'decreasing').replace(2424, 'increasing')
    else:
        pdf['variance'] = 'constant'

    mdf = exp_data['mouselab-mdp'].set_index('pid').loc[complete]

    def extract(q):
        return list(map(int, q['click']['state']['target']))

    mdf['clicks'] = mdf.queries.apply(extract)
    mdf['n_clicks'] = mdf.clicks.apply(len)
    mdf['thinking'] = mdf['rt'].apply(get(0, default=0))
    mdf['variance'] = pdf.variance

    tdf = mdf.query('block == "test"').copy()
    tdf.trial_index -= tdf.trial_index.min()
    tdf.trial_index = tdf.trial_index.astype(int)
    tdf.trial_id = tdf.trial_id.astype(int)

    # pdf['total_time'] = exp_data['survey'].time_elapsed / 60000

    pdf['n_clicks'] = tdf.groupby('pid').n_clicks.mean()
    pdf['score'] = tdf.groupby('pid').score.mean()
    pdf['thinking'] = mdf.groupby('pid').thinking.mean()

    def excluded_pids():
        sdf = exp_data['survey-multi-choice'].set_index('pid').loc[complete]
        responses = pd.DataFrame(list(sdf.responses), index=sdf.index)
        grp = responses.groupby(lambda pid: pdf.variance[pid])
        correct = grp.apply(lambda x: x.mode().iloc[0])  # assume the most common answer is correct
        errors = correct.loc[pdf.variance].set_index(pdf.index) != responses
        fail_quiz = errors.sum(1) > 1
        no_click = mdf.query('block == "train_inspector"').groupby('pid').n_clicks.sum() == 0
        return fail_quiz | no_click

    pdf['excluded'] = excluded_pids()
    tdf = tdf.loc[~pdf.excluded]
    print(f'Excluding {pdf.excluded.sum()} out of {len(pdf)} partipicants')

    def get_env(row):
        row.state_rewards[0] = 0
        sigma = 5 if row.variance == 'constant' else 4

        return make_env(0, sigma,
                        scaling_factors=SCALING[row.variance],
                        ground_truth=row.state_rewards)
    tdf['env'] = tdf.apply(get_env, axis=1)

    def unroll(df):
        for pid, row in df.iterrows():
            env = row.env
            env.reset()
            for a in [*row.clicks, env.term_action]:
                yield {'pid': pid, 'trial_index': row.trial_index, 'trial_id': row.trial_id,
                       'state': env._state, 'action': a}
                env.step(a)
    return {
        'participants': pdf,
        'trials': tdf,
        'unrolled': pd.DataFrame(unroll(tdf)),

    }



