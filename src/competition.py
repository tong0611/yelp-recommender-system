"""
competition.py - entry point of the Yelp rating prediction pipeline:
a staged Spark RDD ETL pipeline feeding an XGBoost ensemble.

Usage:
    spark-submit src/competition.py <folder_path> <test_file> <output_file> [feature_cache_dir]

Modules:
    config.py     constants, feature-block sizes, XGBoost configurations
    utils.py      RunReport (stage timing / metrics), FeatureStore (cache), small helpers
    features.py   every feature definition, pure Python (entity- and pair-level)
    etl.py        Spark stages: extract, validate, transform, oof_stats, assemble
    model.py      XGBoost ensemble training and evaluation

Pipeline (every stage is timed and its record counts are logged in a run report):
    1. extract    parse raw CSV/JSON sources with fault-tolerant parsing (bad lines are counted,
                  not fatal)
    2. validate   data-quality checks: rating range, duplicate pairs, orphan ids, train/test overlap
                  (leakage) and cold-start coverage; deterministic fold assignment
    3. transform  entity tables (user, business, market context, check-in time profile, photos,
                  tip sentiment from a word+bigram lexicon learned on review text); optionally kept
                  in a feature cache keyed by an input fingerprint
    4. oof_stats  out-of-fold user / business rating statistics (no target leakage)
    5. assemble   ~207 features per pair, built on the executors and streamed to the driver as
                  float32 blocks; friends' ratings via a broadcast (map-side) join
    6. model      average of three XGBoost models (histogram trees)
    7. write / evaluate   predictions CSV; RMSE and error distribution when the test file is labelled

Measured results (yelp_val.csv, local 18-core machine, spark-submit default 1GB driver):
    RMSE 0.95766 (global-mean baseline 1.1222, -14.7%)
    Error distribution: <1: 104343 | 1-2: 31073 | 2-3: 5833 | 3-4: 793 | >=4: 2
    Total ~133s (2 cores: 261s). See README.md for the ablation study and engineering notes.
"""
import os
import sys

import numpy as np
from pyspark import SparkConf, SparkContext, StorageLevel

from config import N_CTX, N_STAT, N_USER_F
from etl import assemble, assign_folds, build_context, build_entities, parse_ratings, rating_stats, validate
from model import evaluate, train_and_predict
from utils import FeatureStore, RunReport

# Project modules shipped to the executors. Functions from imported modules are pickled by
# reference (module + name), so each Python worker must be able to import these files.
MODULES = ('config', 'utils', 'features', 'etl', 'model')
# Raw inputs whose sizes / mtimes make up the feature-cache fingerprint.
INPUT_FILES = ('yelp_train.csv', 'user.json', 'business.json', 'photo.json', 'checkin.json',
               'tip.json', 'review_train.json')


def main():
    """Run the pipeline end to end and print the run report."""
    folder, test_file, out_file = sys.argv[1], sys.argv[2], sys.argv[3]
    cache_dir = sys.argv[4] if len(sys.argv) > 4 else None
    report = RunReport()
    sc = SparkContext(conf=SparkConf().setAppName('yelp-recsys').setMaster('local[*]'))
    sc.setLogLevel('ERROR')
    here = os.path.dirname(os.path.abspath(__file__))
    for m in MODULES:
        sc.addPyFile(os.path.join(here, m + '.py'))
    n_parts = max(16, 2 * sc.defaultParallelism)   # textFile gives small CSVs only 2 partitions
    bad_json = sc.accumulator(0)   # malformed lines, counted across executors
    bad_csv = sc.accumulator(0)

    with report.stage('extract'):
        # Lazy: the files are actually read by the first action in the validate stage.
        train_raw = parse_ratings(sc, folder + '/yelp_train.csv', bad_csv)
        test = parse_ratings(sc, test_file, bad_csv).persist(StorageLevel.MEMORY_AND_DISK)

    with report.stage('validate'):
        clean = validate(train_raw, test, report)
        train = assign_folds(clean).persist(StorageLevel.MEMORY_AND_DISK)
        report.metric('validate', 'train rows after cleaning', train.count())
        # Ids of this run, broadcast so the big JSON files can be filtered down to them.
        users = sc.broadcast(set(train.map(lambda x: x[0]).union(test.map(lambda x: x[0])).distinct().collect()))
        biz = sc.broadcast(set(train.map(lambda x: x[1]).union(test.map(lambda x: x[1])).distinct().collect()))
        mean_rating = train.map(lambda x: x[2]).mean()

    with report.stage('transform'):
        inputs = [os.path.join(folder, f) for f in INPUT_FILES] + [test_file]
        store = FeatureStore(sc, cache_dir, inputs)
        side, status = store.load_or_build(
            'entities', lambda: build_entities(sc, folder, users, biz, mean_rating, bad_json))
        report.metric('transform', 'entity tables', status)
        report.metric('transform', 'users / businesses with profiles',
                      '%d / %d' % (len(side['user']), len(side['biz'])))
        report.metric('transform', 'sentiment lexicon tokens', len(side['lexicon']))
        report.metric('transform', 'users with friends in the data', len(side['friends']))
        user_avg = sc.broadcast(dict((u, f[1]) for u, f in side['user'].items()))
        biz_stars = sc.broadcast(dict((b, f[0]) for b, f in side['biz'].items()))

    with report.stage('oof_stats'):
        # The lexicon is only needed here, so it is popped to keep it out of the broadcast context.
        stats = rating_stats(sc, folder, train, user_avg, biz_stars, side.pop('lexicon'), bad_json)

        # Accumulators are only reliable after the actions that update them have run.
        report.metric('extract', 'malformed JSON lines skipped', bad_json.value)
        report.metric('extract', 'malformed CSV rows skipped', bad_csv.value)
        report.metric('validate', 'users missing from user.json',
                      len([u for u in users.value if u not in side['user']]))
        report.metric('validate', 'businesses missing from business.json',
                      len([b for b in biz.value if b not in side['biz']]))

    with report.stage('assemble'):
        hist, rated = build_context(train, biz_stars)
        friends = side.pop('friends')
        n_bf = side['n_bf']
        report.metric('assemble', 'friend edges (broadcast)', sum(len(v) for v in friends.values()))
        # One broadcast holding everything pair_features needs; shipped once per executor.
        ctx = sc.broadcast(dict(side=side, stats=stats, hist=hist, rated=rated, friends=friends,
                                mu=mean_rating))
        del side, stats, hist, rated, friends   # driver copies no longer needed once broadcast
        x_train, y_train = assemble(train, ctx, True, n_parts)
        x_test, keys = assemble(test.map(lambda x: (x[0], x[1], None, None)), ctx, False, n_parts)
        # Social coverage: the social block starts after user, business, market and two stat blocks.
        soc_col = N_USER_F + n_bf + N_CTX + 2 * N_STAT
        report.metric('assemble', 'rows with a friend rating (train / test)', '%.1f%% / %.1f%%' % (
            100.0 * (x_train[:, soc_col] > 0).mean(), 100.0 * (x_test[:, soc_col] > 0).mean()))
        y_train = np.asarray(y_train, dtype=np.float32)
        report.metric('assemble', 'feature matrix (train)', '%d x %d' % x_train.shape)
        report.metric('assemble', 'feature matrix (test)', '%d x %d' % x_test.shape)

    with report.stage('model'):
        pred = train_and_predict(x_train, y_train, x_test)

    with report.stage('write'):
        lines = sc.parallelize(list(zip(keys, pred.tolist()))) \
            .map(lambda kv: kv[0] + ',' + repr(kv[1])).collect()
        with open(out_file, 'w') as f:
            f.write('user_id, business_id, prediction\n')
            f.write('\n'.join(lines) + '\n')

    with report.stage('evaluate'):
        result = evaluate(sc, test, keys, pred)
        if result:                     # None when the test file has no labels
            rmse, bins = result
            report.metric('evaluate', 'RMSE', '%.5f' % rmse)
            for lab, v in zip(['>=0 and <1', '>=1 and <2', '>=2 and <3', '>=3 and <4', '>=4'], bins):
                report.metric('evaluate', 'error ' + lab, v)

    print(report.render())
    sc.stop()


if __name__ == '__main__':
    main()
