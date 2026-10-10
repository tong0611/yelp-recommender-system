"""
etl.py - the Spark (RDD-only) stages of the pipeline.

    extract    parse_ratings        rating CSV -> RDD, malformed rows counted not fatal
    validate   validate             data-quality checks, de-duplication
               assign_folds         deterministic K-fold assignment
    transform  build_entities       user / business / market / photo / check-in / tip tables,
                                    review-learned sentiment lexicon
    oof_stats  rating_stats         per-fold rating statistics (leak-free target features)
    assemble   build_context        user histories and per-rater ratings for the broadcast context
               assemble             feature matrices built on executors, streamed as float32 blocks

The per-row feature logic lives in features.py; this module only moves and aggregates data.
"""
import numpy as np
from pyspark import StorageLevel

from config import K_FOLDS, LEX_ALPHA, LEX_MIN_COUNT, N_TOP_CATS, NAN, PHOTO_LABELS
from features import (business_features, checkin_profile, market_context, pair_features,
                      review_vector, user_features)
from utils import add_vec, chain_key, safe_json, split_cats, tokens


# ============================================================================ stage: extract
def parse_ratings(sc, path, bad):
    """Rating CSV -> RDD of (user, business, stars or None). Malformed rows are counted, not fatal;
    stars is None when the file has no rating column (e.g. an unlabeled test file)."""
    lines = sc.textFile(path)
    header = lines.first()

    def parse(line):
        p = line.strip().split(',')
        if len(p) < 2 or not p[0] or not p[1]:
            bad.add(1)
            return []
        stars = None
        if len(p) > 2:
            try:
                stars = float(p[2])
            except ValueError:
                stars = None
        return [(p[0], p[1], stars)]

    return lines.filter(lambda l: l != header).flatMap(parse)


# ============================================================================ stage: validate
def validate(train_raw, test, report):
    """Data-quality checks on the rating files; returns the cleaned (valid, de-duplicated) train RDD.
    Records: invalid ratings, duplicate pairs, train/test overlap (a leakage check) and how many test
    rows involve users / businesses never seen in training (cold start)."""
    bad_rating = train_raw.filter(lambda x: x[2] is None or not 1.0 <= x[2] <= 5.0).count()
    dup_pairs = train_raw.map(lambda x: ((x[0], x[1]), 1)).reduceByKey(lambda a, b: a + b) \
        .filter(lambda kv: kv[1] > 1).count()
    clean = train_raw.filter(lambda x: x[2] is not None and 1.0 <= x[2] <= 5.0) \
        .map(lambda x: ((x[0], x[1]), x[2])).reduceByKey(lambda a, b: a) \
        .map(lambda kv: (kv[0][0], kv[0][1], kv[1]))   # keep one rating per (user, business)
    # A pair in both train and test would let the model memorise the answer.
    overlap = clean.map(lambda x: ((x[0], x[1]), 1)) \
        .join(test.map(lambda x: ((x[0], x[1]), 1))).count()
    train_users = clean.map(lambda x: (x[0], 1)).distinct()
    train_biz = clean.map(lambda x: (x[1], 1)).distinct()
    cold_u = test.map(lambda x: (x[0], 1)).subtractByKey(train_users).count()
    cold_b = test.map(lambda x: (x[1], 1)).subtractByKey(train_biz).count()
    report.metric('validate', 'train rows with invalid rating', bad_rating)
    report.metric('validate', 'duplicate train pairs (deduped)', dup_pairs)
    report.metric('validate', 'train/test overlapping pairs', overlap)
    report.metric('validate', 'test rows', test.count())
    report.metric('validate', 'test rows with unseen user', cold_u)
    report.metric('validate', 'test rows with unseen business', cold_b)
    return clean


def assign_folds(clean):
    """(user, business, stars) -> (user, business, stars, fold).

    Folds come from a stable sort order plus the row index, not from Python's hash(), which is
    salted per process (PYTHONHASHSEED) and would give different folds on every run."""
    return clean.sortBy(lambda x: (x[0], x[1])).zipWithIndex() \
        .map(lambda x: (x[0][0], x[0][1], x[0][2], x[1] % K_FOLDS))


# ============================================================================ stage: transform
def build_entities(sc, folder, users, biz, mean_rating, bad):
    """Build every side table as a dict keyed by id, using RDD aggregations.

    users, biz: broadcast sets of the ids that appear in train or test; other ids are skipped."""
    # user.json is 2GB. Every line starts with '{"user_id":"<22 chars>"', so the id sits at a fixed
    # offset and lines can be filtered with a cheap substring check before the costly json.loads
    # (skips >99% of the parsing).
    user_raw = sc.textFile(folder + '/user.json') \
        .filter(lambda l: l[12:34] in users.value) \
        .map(lambda l: safe_json(l, bad)).filter(lambda d: d is not None) \
        .persist(StorageLevel.MEMORY_AND_DISK)   # used for profiles and for the friend graph
    user_f = user_raw.map(lambda d: (d['user_id'], user_features(d))).collectAsMap()
    # Friend graph restricted to users of this run: user -> [friends who also appear in the data].
    friends = user_raw.map(lambda d: (d['user_id'], [
        f.strip() for f in (d.get('friends') or 'None').split(',')
        if f.strip() in users.value])).filter(lambda kv: kv[1]).collectAsMap()
    user_raw.unpersist()

    # Market aggregates over ALL businesses (not only ours): chains, cities, postal codes, categories.
    biz_all = sc.textFile(folder + '/business.json').map(lambda l: safe_json(l, bad)) \
        .filter(lambda d: d is not None and d.get('stars') is not None) \
        .persist(StorageLevel.MEMORY_AND_DISK)

    def market_agg(key_fn):
        """key -> [#businesses, sum stars, sum review_count] over every business with that key."""
        return biz_all.flatMap(lambda d: [(k, [1, d['stars'], d.get('review_count', 0)]) for k in key_fn(d)]) \
            .reduceByKey(add_vec).collectAsMap()

    chain = market_agg(lambda d: [chain_key(d)] if chain_key(d) else [])
    city = market_agg(lambda d: [(d.get('city') or '', d.get('state') or '')])
    postal = market_agg(lambda d: [d['postal_code']] if d.get('postal_code') else [])
    cat = market_agg(split_cats)

    ours = biz_all.filter(lambda d: d['business_id'] in biz.value) \
        .persist(StorageLevel.MEMORY_AND_DISK)   # reused by the four aggregations below
    biz_all.unpersist()
    top_cats = [c for c, _ in ours.flatMap(lambda d: [(c, 1) for c in split_cats(d)])
                .reduceByKey(lambda x, y: x + y)
                .takeOrdered(N_TOP_CATS, key=lambda x: (-x[1], x[0]))]   # ties by name: deterministic
    biz_f = ours.map(lambda d: (d['business_id'], business_features(d, top_cats))).collectAsMap()
    biz_meta = ours.map(lambda d: (d['business_id'],
                                   (frozenset(split_cats(d)), d.get('city') or ''))).collectAsMap()
    mk = sc.broadcast((chain, city, postal, cat))
    biz_ctx = ours.map(lambda d: (d['business_id'], market_context(d, *mk.value))).collectAsMap()
    mk.unpersist()
    ours.unpersist()

    # business -> [total photos, #food, #inside, #outside, #drink, #menu]
    photo = sc.textFile(folder + '/photo.json').map(lambda l: safe_json(l, bad)) \
        .filter(lambda d: d is not None and d.get('business_id') in biz.value) \
        .map(lambda d: (d['business_id'], [1] + [1 if d.get('label') == l else 0 for l in PHOTO_LABELS])) \
        .reduceByKey(add_vec).collectAsMap()

    checkin = sc.textFile(folder + '/checkin.json').map(lambda l: safe_json(l, bad)) \
        .filter(lambda d: d is not None and d.get('business_id') in biz.value) \
        .map(lambda d: (d['business_id'], checkin_profile(d.get('time') or {}))).collectAsMap()

    # Sentiment lexicon from training review text: each token's score is the smoothed mean of
    # (review stars - global mean) over the reviews containing it, e.g. "gem" +0.71, "rudest" -1.75.
    # Learned on reviews (which carry stars), applied to tips (which do not).
    lexicon = sc.textFile(folder + '/review_train.json').map(lambda l: safe_json(l, bad)) \
        .filter(lambda d: d is not None) \
        .flatMap(lambda d: [(t, (d['stars'] - mean_rating, 1)) for t in tokens(d.get('text', ''))]) \
        .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])) \
        .filter(lambda x: x[1][1] >= LEX_MIN_COUNT) \
        .mapValues(lambda v: v[0] / (v[1] + LEX_ALPHA)).collectAsMap()
    lex = sc.broadcast(lexicon)

    def tip_record(d):
        """(user, business, [sum of token scores, #scored tokens, 1 tip, likes]) for one tip."""
        scores = [lex.value[t] for t in tokens(d.get('text', '')) if t in lex.value]
        return d['user_id'], d['business_id'], (sum(scores), len(scores), 1, d.get('likes', 0))

    tips = sc.textFile(folder + '/tip.json').map(lambda l: safe_json(l, bad)) \
        .filter(lambda d: d is not None and (d['user_id'] in users.value or d['business_id'] in biz.value)) \
        .map(tip_record).persist(StorageLevel.MEMORY_AND_DISK)   # aggregated three ways below
    tip_pair = tips.filter(lambda t: t[0] in users.value and t[1] in biz.value) \
        .map(lambda t: ((t[0], t[1]), t[2])).reduceByKey(add_vec).collectAsMap()
    tip_user = tips.filter(lambda t: t[0] in users.value) \
        .map(lambda t: (t[0], t[2])).reduceByKey(add_vec).collectAsMap()
    tip_biz = tips.filter(lambda t: t[1] in biz.value) \
        .map(lambda t: (t[1], t[2])).reduceByKey(add_vec).collectAsMap()
    tips.unpersist()
    lex.unpersist()

    return dict(user=user_f, friends=friends, biz=biz_f, biz_meta=biz_meta, biz_ctx=biz_ctx,
                n_bf=len(business_features({}, top_cats)),   # business vector length (for missing ids)
                photo=photo, checkin=checkin, tip_pair=tip_pair, tip_user=tip_user, tip_biz=tip_biz,
                lexicon=lexicon)


# ============================================================================ stage: oof_stats
def rating_stats(sc, folder, train, user_avg, biz_stars, lexicon, bad):
    """Per user and per business sums of review_vector, overall and per fold.

    Returns stats['u'|'b'][fold][id] -> summed vector, where index K_FOLDS holds the totals. A
    training row in fold k later uses (total - fold k): statistics from the other folds only, so its
    own rating never leaks into its features (plain or leave-one-out averages overfit badly here:
    with leave-one-out, val RMSE got worse with every boosting round, 0.971 -> 0.981)."""
    lex = sc.broadcast(lexicon)

    def review_extra(d):
        """(user, business) -> (useful, funny, cool, text length, year, mean token sentiment)."""
        scores = [lex.value[t] for t in tokens(d.get('text', '')) if t in lex.value]
        sent = sum(scores) / len(scores) if scores else 0.0
        year = int((d.get('date') or '2018')[:4])
        return ((d['user_id'], d['business_id']),
                (d.get('useful', 0), d.get('funny', 0), d.get('cool', 0), len(d.get('text', '')),
                 year, sent))

    extra = sc.textFile(folder + '/review_train.json').map(lambda l: safe_json(l, bad)) \
        .filter(lambda d: d is not None).map(review_extra)

    def emit(rec):
        """Send one review's vector to its user and business, in its own fold and in the totals."""
        (u, b), ((r, fold), rv) = rec
        rv = rv or (0, 0, 0, 0, 2018, 0.0)   # rating without review text: neutral defaults
        vec = review_vector(r, biz_stars.value.get(b, NAN), user_avg.value.get(u, NAN), rv)
        return [(('u', u, fold), vec), (('u', u, -1), vec), (('b', b, fold), vec), (('b', b, -1), vec)]

    sums = train.map(lambda x: ((x[0], x[1]), (x[2], x[3]))).leftOuterJoin(extra) \
        .flatMap(emit).reduceByKey(add_vec).collect()
    lex.unpersist()
    stats = {'u': [{} for _ in range(K_FOLDS + 1)], 'b': [{} for _ in range(K_FOLDS + 1)]}
    for (kind, key, fold), vec in sums:
        stats[kind][fold][key] = vec   # fold -1 indexes the last slot, i.e. K_FOLDS (the totals)
    return stats


# ============================================================================ stage: assemble
def build_context(train, biz_stars):
    """Two per-user lookup tables built from the training ratings, later broadcast:

    hist:  user -> [(business, rating - business stars, fold)], for history_affinity
    rated: rater -> {business: stars}, for friends' ratings (map-side join in social_lookup)"""
    hist = train.map(lambda x: (x[0], (x[1], x[2] - biz_stars.value.get(x[1], NAN), x[3]))) \
        .groupByKey().mapValues(list).collectAsMap()
    rated = train.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(dict).collectAsMap()
    return hist, rated


def assemble(rows, ctx_bc, with_label, n_parts):
    """Build feature matrices on the executors.

    rows: RDD of (user, business, label/None, fold/None).
    Each partition is packed into one float32 block and pulled with toLocalIterator (one partition
    at a time), so the driver JVM (1GB by default under spark-submit) never holds the whole matrix.
    Returns (X, labels) for training rows or (X, "user,business" keys) for test rows."""
    keyed = rows.repartition(n_parts)   # spread rows over all cores

    def part(it):
        c = ctx_bc.value                # deserialised once per partition, not once per row
        feats, extra = [], []
        for u, b, label, fold in it:
            feats.append(pair_features(u, b, fold, c))
            extra.append(label if with_label else u + ',' + b)
        if feats:
            # float32 numpy is ~10x smaller than a list of Python floats.
            yield np.asarray(feats, dtype=np.float32), extra

    xs, extras = [], []
    for x, e in keyed.mapPartitions(part).toLocalIterator():
        xs.append(x)
        extras.extend(e)
    return np.vstack(xs), extras
