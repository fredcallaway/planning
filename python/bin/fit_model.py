#!/usr/bin/env python3

import os
import sys
sys.path.append('.')  # shouldn't be necessary, but it is on PNI's servers

from pandas import Series, DataFrame
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from models.classical import *
from models.optimal import OptimalModel
from models.utils import make_env, fetch_data
from models.error import ErrorModel

EXP = 1
DATA = None  # we make this a global variable for multiprocessing efficiency
ENV = make_env(0, 4)

ERROR_PARAMS = {
    'p_error': np.linspace(0.01,0.25, 25),
    'temp': np.logspace(-5,1, 50),
}
PATHS = {
    'individual': 'results/mle/individual/',
    'aggregate': 'results/mle/aggregate'
}
MODELS = [
    # BestFirstModel,
    OptimalModel,
]

def get_mle(model, data):
    err_mod = ErrorModel(ENV, model.preference, data)
    mle = Series(err_mod.maximum_likelihood_estimate(**ERROR_PARAMS))
    mle['model'] = type(model).__name__
    mle['n'] = len(data)
    for k, v in model.__dict__.items():
        mle[k] = v
    return Series(mle)

def write_mle(model, data=None):
    if data is None:
        data = DATA
    get_mle(model, data).to_pickle(os.path.join(PATHS['aggregate'], str(model)))

    def fit_individuals():
        for pid, dd in data.groupby('pid'):
            mle = get_mle(model, dd)
            mle['pid'] = pid
            yield mle

    DataFrame(fit_individuals()).to_pickle(os.path.join(PATHS['individual'], str(model)))


    


def main():
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)
    jobs = [delayed(write_mle)(mod) 
            for model in MODELS
            for mod in model.all_models(ENV)][:3]
    njob = min(os.cpu_count() - 2, len(jobs))
    Parallel(njob, verbose=10)(jobs)

if __name__ == '__main__':
    DATA = fetch_data(EXP)['unrolled']
    main()
