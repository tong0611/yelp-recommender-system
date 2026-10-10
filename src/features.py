"""
features.py - every feature definition, as pure Python functions (no Spark).

Keeping feature logic free of Spark makes it testable on plain dicts and lets the same functions run
on the driver or inside executors. Two levels:

    entity features   one vector per user / business, from the raw JSON records
                      (user_features, business_features, checkin_profile, market_context)
    pair features     one vector per (user, business) row, combining entity tables, out-of-fold
                      rating statistics, friends' ratings and the user's history (pair_features)

Final row layout (pair_features):
    user | business | market context | user OOF stats | business OOF stats | social |
    history affinity | tip counts | tip sentiment | photos | check-ins | crosses
"""
import math

from config import (ATTR_MAP, BOOL_ATTRS, COMPLIMENTS, DICT_ATTRS, K_FOLDS, N_CHECKIN, N_CTX,
                    N_PHOTO, N_SOCIAL, N_STAT, N_USER_F, NAN, RECENT_YEAR, STARS_PRIOR)
from utils import chain_key, clean_attr, haversine_km, ratio, split_cats


# ============================================================================ entity features
def user_features(d):
    """Numeric profile of a user from user.json. Index 1 is average_stars (used elsewhere).

    [review_count, average_stars, useful, funny, cool, fans, #friends, #elite years,
     months on Yelp, total compliments, useful/review, fans/review, compliments/review,
     (funny+cool)/review, last elite year] + 11 compliment counts"""
    since = d.get('yelping_since') or '2018-01-01'
    friends = d.get('friends') or 'None'
    elite = d.get('elite') or 'None'
    elite_years = [int(y) for y in elite.split(',') if y.strip().isdigit()] if elite != 'None' else []
    comps = [d.get('compliment_' + c, 0) for c in COMPLIMENTS]
    rc = d.get('review_count', 0)
    return [rc, d.get('average_stars', NAN), d.get('useful', 0), d.get('funny', 0), d.get('cool', 0),
            d.get('fans', 0),
            0 if friends == 'None' else friends.count(',') + 1,   # friend list is one long string
            len(elite_years),
            (2019 - int(since[:4])) * 12 - int(since[5:7]),       # tenure in months (data ends 2019)
            sum(comps),
            # Per-review ratios: activity normalised by how much the user writes.
            ratio(d.get('useful', 0), rc), ratio(d.get('fans', 0), rc), ratio(sum(comps), rc),
            ratio(d.get('funny', 0) + d.get('cool', 0), rc),
            max(elite_years) if elite_years else NAN] + comps


def business_features(d, top_cats):
    """Numeric profile of a business from business.json.
    Index 0 = stars, 1 = review_count, 3/4 = latitude/longitude, 5 = price tier (used elsewhere)."""
    a = d.get('attributes') or {}
    lat, lon = d.get('latitude'), d.get('longitude')
    f = [d.get('stars', NAN), d.get('review_count', 0), d.get('is_open', NAN),
         NAN if lat is None else lat, NAN if lon is None else lon]
    pr = a.get('RestaurantsPriceRange2')
    f.append(float(pr) if pr and pr.isdigit() else NAN)
    for k in sorted(ATTR_MAP):          # sorted: fixed column order across runs
        v = a.get(k)
        f.append(ATTR_MAP[k].get(clean_attr(v), NAN) if v else NAN)
    for k in BOOL_ATTRS:
        v = a.get(k)
        f.append(NAN if v is None else (1.0 if v == 'True' else 0.0))
    for k in DICT_ATTRS:
        v = a.get(k)
        f.append(NAN if v is None else float(v.count('True')))
    cats = set(split_cats(d))
    f.extend([float(len(a)), float(len(cats))])
    f.extend(1.0 if c in cats else 0.0 for c in top_cats)   # one-hot of the top categories
    # Opening hours: number of open days and total open hours per week ("8:30-17:0").
    hours = d.get('hours') or {}
    total = 0.0
    for v in hours.values():
        try:
            o, c = v.split('-')
            oh, om = o.split(':')
            ch, cm = c.split(':')
            span = (int(ch) + int(cm) / 60.0) - (int(oh) + int(om) / 60.0)
            total += span if span > 0 else span + 24   # closes after midnight
        except ValueError:
            pass                                       # malformed entry: skip that day
    f.extend([float(len(hours)), total if hours else NAN])
    return f


def checkin_profile(times):
    """When a business is visited: total check-ins, active weekday-hour slots, and the share of
    check-ins on weekends, in the morning (6-10h), in the evening (17-22h) and late (23-4h)."""
    total = float(sum(times.values()))
    if not total:
        return [0.0, 0.0, NAN, NAN, NAN, NAN]
    wknd = morning = evening = night = 0
    for k, v in times.items():
        day, hour = k.split('-')     # keys look like "Fri-21"
        hour = int(hour)
        if day in ('Sat', 'Sun'):
            wknd += v
        if 6 <= hour <= 10:
            morning += v
        elif 17 <= hour <= 22:
            evening += v
        elif hour >= 23 or hour <= 4:
            night += v
    return [total, float(len(times)), wknd / total, morning / total, evening / total, night / total]


def market_context(d, chain, city, postal, cat):
    """How a business compares with its market, using aggregates over all of business.json.

    chain/city/postal/cat map a key to [#businesses, sum of stars, sum of review counts].
    Returns [chain size, chain mean stars at *other* locations, city size, city mean stars,
             postal-code size, postal-code mean stars, category mean stars,
             stars - city mean, stars - category mean, stars - chain mean, shrunk stars]."""
    stars = d.get('stars', NAN)
    ck = chain.get(chain_key(d), [1, stars, 0])
    n_chain = ck[0]
    chain_other = (ck[1] - stars) / (n_chain - 1) if n_chain > 1 else NAN   # exclude this location
    cs = city.get((d.get('city') or '', d.get('state') or ''))
    ps = postal.get(d.get('postal_code') or '')
    city_mean = cs[1] / cs[0] if cs else NAN
    postal_mean = ps[1] / ps[0] if ps else NAN
    # Category mean: average of the means of every category the business belongs to.
    cm = [cat[c][1] / cat[c][0] for c in split_cats(d) if c in cat]
    cat_mean = sum(cm) / len(cm) if cm else NAN
    # Bayesian shrinkage: few reviews -> pulled toward the category mean.
    rc = d.get('review_count', 0)
    prior = cat_mean if cat_mean == cat_mean else 3.7   # cat_mean == cat_mean is False for NaN
    shrunk = (stars * rc + prior * STARS_PRIOR) / (rc + STARS_PRIOR)
    return [float(n_chain), chain_other, float(cs[0]) if cs else NAN, city_mean,
            float(ps[0]) if ps else NAN, postal_mean, cat_mean,
            stars - city_mean, stars - cat_mean, stars - chain_other, shrunk]


# ============================================================================ out-of-fold statistics
def review_vector(r, bs, ua, rv):
    """Additive statistics of one training review (r = stars, bs = business stars, ua = user's
    average stars, rv = (useful, funny, cool, text length, year, text sentiment)):
    [1, r, r^2, r - bs, r - ua, useful, funny, cool, text length, year, is_recent, r * is_recent,
     sentiment]

    Everything is a sum, so vectors can be added with reduceByKey and a fold's contribution can be
    subtracted again later (see agg_stats)."""
    recent = 1.0 if rv[4] >= RECENT_YEAR else 0.0
    return [1.0, r, r * r, r - bs, r - ua, rv[0], rv[1], rv[2], rv[3], rv[4], recent, r * recent, rv[5]]


def agg_stats(s, sub):
    """Summed review vectors -> [n, mean, std, bias vs business stars, bias vs user average,
    mean useful, funny, cool, text length, mean year, recent share, recent mean - overall mean,
    mean review sentiment], after subtracting `sub` (the row's own fold) when given."""
    empty = [0.0] + [NAN] * (N_STAT - 1)
    if s is None:
        return empty
    if sub is not None:
        s = [x - y for x, y in zip(s, sub)]   # out-of-fold: drop the row's own fold
    n = s[0]
    if n < 0.5:                       # nothing left once the row's own fold is removed
        return empty
    mean = s[1] / n
    std = math.sqrt(max(s[2] / n - mean * mean, 0.0))      # Var = E[r^2] - E[r]^2
    trend = s[11] / s[10] - mean if s[10] > 0.5 else NAN   # recent mean minus overall mean
    return [n, mean, std] + [x / n for x in s[3:10]] + [s[10] / n, trend, s[12] / n]


# ============================================================================ social ratings
def social_lookup(u, b, c):
    """Ratings the user's friends gave the target business, as a map-side (broadcast) join.

    c['friends']: user -> friends who appear in the data; c['rated']: rater -> {business: stars}.
    Returns [#friends who rated b, sum of their ratings, sum of (rating - business stars),
    sum of (rating - that friend's own average)], or None when no friend rated b.

    This replaced a shuffle join (friend edges x every rating each friend made, then filtered to
    the needed pairs) that generated tens of millions of intermediate records; both tables are small
    enough to broadcast, so each row needs only ~#friends dictionary lookups and nothing is shuffled.
    A friend's rating is a different row, so the target rating itself is never used."""
    rated, side = c['rated'], c['side']
    n = sr = sb = sf = 0.0
    bs = side['biz'].get(b)
    bs = bs[0] if bs else NAN
    for f in c['friends'].get(u, ()):
        r = rated.get(f, {}).get(b)
        if r is None:
            continue
        fa = side['user'].get(f)
        n += 1
        sr += r
        sb += r - bs
        sf += r - (fa[1] if fa else NAN)
    return [n, sr, sb, sf] if n else None


def social_features(s):
    """[#friends who rated it, their mean rating, mean deviation from business stars,
    mean deviation from their own averages]; empty -> zeros / NaN."""
    if s is None:
        return [0.0] + [NAN] * (N_SOCIAL - 1)
    n = s[0]
    return [n, s[1] / n, s[2] / n, s[3] / n]


# ============================================================================ pair features
def sentiment(s):
    """[mean score per scored token, mean score per tip] from an aggregated tip vector
    (sum of token scores, #scored tokens, #tips, likes)."""
    if s is None:
        return [NAN, NAN]
    return [s[0] / s[1] if s[1] else NAN, s[0] / s[2]]


def history_affinity(u, b, fold, c):
    """Affinities from the user's training history, always excluding the target business:
    category-similarity-weighted residuals (out-of-fold only), distance from the user's activity
    centroid, spread of that activity, price-tier gap, new-city flag and number of cities."""
    side = c['side']
    meta, bf = side['biz_meta'], side['biz']
    tm = meta.get(b)
    cb = tm[0] if tm else None          # target business's category set
    tf = bf.get(b)
    ws = wr = s3 = c3 = mx = 0.0
    n_aff = 0
    slat = slon = slat2 = slon2 = ng = sp = n_p = 0.0
    cities = set()
    for b2, res, f2 in c['hist'].get(u, ()):        # res = rating - business stars
        if b2 == b:
            continue
        f2v = bf.get(b2)
        m2 = meta.get(b2)
        # Geography and price use only business attributes (no ratings), so all folds are fine.
        if f2v is not None:
            lat, lon, price = f2v[3], f2v[4], f2v[5]
            if lat == lat and lon == lon:      # x == x is False only for NaN
                slat += lat
                slon += lon
                slat2 += lat * lat
                slon2 += lon * lon
                ng += 1
            if price == price:
                sp += price
                n_p += 1
        if m2:
            cities.add(m2[1])
        # Residual-based affinity uses only other folds (the residual contains a rating).
        if f2 == fold or not cb or not m2 or not m2[0]:
            continue
        inter = len(cb & m2[0])
        if not inter:
            continue
        j = inter / float(len(cb | m2[0]))      # Jaccard similarity of the category sets
        n_aff += 1
        ws += j
        wr += j * res
        mx = max(mx, j)
        if j >= 0.3:                            # "closely similar" businesses
            s3 += res
            c3 += 1
    aff = [wr / ws if ws else NAN, ws, n_aff, mx, s3 / c3 if c3 else NAN, c3]
    dist = spread = price_gap = NAN
    if ng and tf is not None and tf[3] == tf[3] and tf[4] == tf[4]:
        clat, clon = slat / ng, slon / ng
        dist = haversine_km(clat, clon, tf[3], tf[4])
        # Rough radius of the user's activity in km (1 degree ~ 111 km).
        spread = 111.0 * math.sqrt(max(slat2 / ng - clat * clat, 0) + max(slon2 / ng - clon * clon, 0))
    if n_p and tf is not None and tf[5] == tf[5]:
        price_gap = tf[5] - sp / n_p            # > 0: pricier than what the user usually visits
    new_city = NAN if not tm or not cities else (0.0 if tm[1] in cities else 1.0)
    return aff + [dist, spread, price_gap, new_city, float(len(cities))]


def pair_features(u, b, fold, c):
    """Full feature vector of one (user, business) pair.

    fold:   the row's fold for training rows (its fold is excluded from rating statistics),
            None for test rows (which use statistics over the whole training set).
    c:      broadcast context with side tables, rating stats, user histories, friend lists,
            per-user ratings and the global mean."""
    side, st = c['side'], c['stats']
    # Missing entities get a NaN block of the right length so every row has the same width.
    uf = side['user'].get(u) or [NAN] * N_USER_F
    bf = side['biz'].get(b) or [NAN] * side['n_bf']
    ctx = side['biz_ctx'].get(b) or [NAN] * N_CTX
    # st[kind][K_FOLDS] = totals, st[kind][fold] = that fold's share (subtracted for training rows).
    us = agg_stats(st['u'][K_FOLDS].get(u), st['u'][fold].get(u) if fold is not None else None)
    bs = agg_stats(st['b'][K_FOLDS].get(b), st['b'][fold].get(b) if fold is not None else None)
    tp = side['tip_pair'].get((u, b))
    tu = side['tip_user'].get(u)
    tb = side['tip_biz'].get(b)
    ph = side['photo'].get(b) or [0] * N_PHOTO
    ck = side['checkin'].get(b) or [0.0, 0.0] + [NAN] * (N_CHECKIN - 2)
    u_avg, b_stars, mu = uf[1], bf[0], c['mu']
    crosses = [u_avg + b_stars - mu,        # classic baseline predictor
               u_avg - b_stars,
               b_stars + us[3],             # business stars + user's out-of-fold bias vs stars
               u_avg + bs[4]]               # user average + business's out-of-fold bias vs users
    return (uf + bf + ctx + us + bs
            + social_features(social_lookup(u, b, c))
            + history_affinity(u, b, fold, c)
            + [tp[2] if tp else 0,                       # tips this user left on this business
               tu[2] if tu else 0, tu[3] if tu else 0,   # user's tip count and likes
               tb[2] if tb else 0, tb[3] if tb else 0]   # business's tip count and likes
            + sentiment(tp) + sentiment(tu) + sentiment(tb)
            + list(ph) + list(ck) + crosses)
