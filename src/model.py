"""
model.py - model training and evaluation.

    train_and_predict   average of the three XGBoost configurations in config.MODELS
    evaluate            RMSE and error distribution against the labelled test file
"""
import math

import numpy as np
import xgboost as xgb

from config import MODELS


def train_and_predict(x_train, y_train, x_test):
    """Train the three XGBoost models and return their averaged predictions clipped to [1, 5]."""
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test)
    pred = np.zeros(x_test.shape[0])
    for params, rounds in MODELS:
        p = dict(params)
        # base_score is explicit because its default differs between xgboost versions.
        p.update({'tree_method': 'hist', 'seed': 0, 'base_score': float(y_train.mean())})
        # xgboost 0.72 uses 'silent'; 1.x+ removed it in favour of 'verbosity'.
        if xgb.__version__.startswith('0.'):
            p['silent'] = 1
        else:
            p['verbosity'] = 0
        pred += xgb.train(p, dtrain, rounds).predict(dtest)
    return np.clip(pred / len(MODELS), 1.0, 5.0)   # ratings live in [1, 5]


def evaluate(sc, test, keys, pred):
    """RMSE and error distribution (|error| buckets [0,1), [1,2), [2,3), [3,4), >=4), computed with
    an RDD join against the true ratings. Returns None when the test file has no labels."""
    truth = test.filter(lambda x: x[2] is not None).map(lambda x: (x[0] + ',' + x[1], x[2]))
    if truth.isEmpty():
        return None
    errors = sc.parallelize(list(zip(keys, pred.tolist()))).join(truth) \
        .map(lambda kv: abs(kv[1][0] - kv[1][1])).cache()
    n = errors.count()
    rmse = math.sqrt(errors.map(lambda e: e * e).sum() / n)
    bins = errors.map(lambda e: (min(int(e), 4), 1)).reduceByKey(lambda x, y: x + y).collectAsMap()
    return rmse, [bins.get(i, 0) for i in range(5)]
