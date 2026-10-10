# Yelp Rating Prediction: Spark ETL + XGBoost

This project develops a personalized recommendation system that predicts the number of stars (1–5)
a Yelp user would assign to a business they have not yet reviewed. By estimating user preferences
from existing review data, the system can recommend the businesses most likely to match each user's
interests. These predicted ratings serve as a key signal for ranking and personalizing
recommendations, so a platform can suggest relevant restaurants and services instead of showing the
same popular businesses to every user. Beyond improving the user experience, the system can help
businesses increase their visibility, reach potential customers, support targeted advertising, and
grow by connecting them with the users most likely to be interested in their services.

The system is a hybrid recommender. It combines collaborative signals (how the user and the
business have rated and been rated, and how the user's friends rated the business) with content
signals (business attributes, categories, location, opening hours, check-ins, photos, and sentiment
learned from review and tip text). A staged PySpark ETL pipeline, using only the RDD API, turns
about 3 GB of raw Yelp data into 207 features per (user, business) pair. A 3-model XGBoost ensemble
then predicts the rating.

**Result:** validation RMSE **0.95766**. That is 14.7% below the global-mean baseline (1.1222) and
under the course target of 0.98. The full run takes about 133 s on a laptop.

## Data

The data is a subset of the [Yelp Open Dataset](https://www.yelp.com/dataset), about 3 GB in total.
The rating files define the prediction task. The JSON files are side information used to build
features.

| File | Size | Records | Contents |
|---|---|---|---|
| `yelp_train.csv` | 22 MB | 455,854 | Training ratings: `user_id, business_id, stars` |
| `yelp_val.csv` | 7 MB | 142,044 | Validation ratings, same format |
| `user.json` | 2.0 GB | 1.5M | User profiles: review count, average stars, fans, friend list, elite years, compliments |
| `business.json` | 140 MB | 189K | Businesses: location, stars, review count, categories, attributes, opening hours |
| `review_train.json` | 449 MB | 456K | Full review text, date and useful/funny/cool votes for the training ratings |
| `tip.json` | 203 MB | 1.2M | Short tips users left on businesses, with likes |
| `checkin.json` | 50 MB | 157K | Check-in counts per business by weekday and hour |
| `photo.json` | 35 MB | 281K | Photo labels per business (food, inside, outside, drink, menu) |

A few properties of the data shaped the design:

- **Sparse ratings.** The rating files cover 11,270 users and 24,951 businesses, so only about 0.2%
  of possible user–business pairs are rated. Many users have only a few ratings, which makes side information
  and regularization important.
- **Most of the raw data is irrelevant to the task.** user.json lists 1.5M users, but only about
  11K appear in the rating files. The pipeline filters to those ids before parsing JSON.
- **Cold start.** Every validation user appears in training, and only 307 validation rows
  (0.2%) involve a business never seen in training.
- **Social signal.** About 52% of rating pairs have at least one friend of the user who also rated
  the same business.

## Project layout

```
src/
  competition.py   entry point: runs the stages in order and prints the run report
  config.py        constants, feature-block sizes, XGBoost configurations
  utils.py         RunReport (stage timing/metrics), FeatureStore (feature cache), helpers
  features.py      every feature definition, as pure Python functions (no Spark)
  etl.py           Spark stages: extract, validate, transform, oof_stats, assemble
  model.py         XGBoost ensemble training and evaluation
data/              Yelp inputs (yelp_train.csv, yelp_val.csv, user/business/review/tip/checkin/photo JSON)
```

## Running

```bash
pip install -r requirements.txt          # also needs Java for Spark
spark-submit src/competition.py data data/yelp_val.csv predictions.csv
# optional 4th argument: a feature-cache directory (re-runs skip the entity-table build)
spark-submit src/competition.py data data/yelp_val.csv predictions.csv /tmp/feature_cache
```

The output CSV has the columns `user_id, business_id, prediction`. If the test file includes stars,
the run report also prints the RMSE and the error distribution.

## Pipeline

| Stage | What it does |
|---|---|
| extract | Parses the CSV/JSON inputs. Malformed lines are counted in accumulators instead of failing the job. |
| validate | Checks rating range, duplicate pairs, train/test overlap (leakage), cold-start coverage and orphan ids. Assigns deterministic folds. |
| transform | Builds the user, business, market-context, check-in, photo and tip tables, and learns the review sentiment lexicon. These tables can be cached. |
| oof_stats | Computes out-of-fold user and business rating statistics. |
| assemble | Builds features on the executors and streams them to the driver as float32 blocks. |
| model | Averages three XGBoost models (`hist` trees). |
| evaluate | Reports RMSE and the error distribution. |

### Feature groups

- **User profile:** activity, popularity, tenure, compliments, per-review ratios.
- **Business profile:** stars, attributes, top-60 categories, opening hours, location.
- **Market context:** chain size and the chain's stars at other locations; city, postal-code and
  category averages; stars shrunk toward the category mean.
- **Out-of-fold stats:** each user's and business's rating mean, std and bias, review votes, length,
  recency trend and review sentiment. These are computed from the other 4 folds.
- **Social:** ratings the user's friends gave this business (52% of pairs have at least one).
- **History affinity:** category-similarity-weighted residuals, distance from the user's usual
  area, price-tier gap, a new-city flag.
- **Tips / photos / check-ins:** tip counts and lexicon sentiment, photo labels, check-in time profile.
- **Crosses:** a baseline predictor (user avg + business stars − global mean) and variants of it.

## Results

Validation set: 142,044 rows. Local 18-core machine, default 1 GB Spark driver.

- RMSE **0.95766**, against a global-mean baseline of 1.1222. The first version of the
  pipeline, with fewer features, scored 0.96007.
- Error distribution: <1: 104,343 · 1–2: 31,073 · 2–3: 5,833 · 3–4: 793 · ≥4: 2.
- Time per stage: validate 5 s, transform 16 s, oof_stats 6 s, assemble 63 s, model 43 s, total about 135 s.
  On 2 cores the total is 261 s.

### Ablation (single XGBoost model, validation RMSE)

| Variant | RMSE |
|---|---|
| all features | 0.95817 |
| without social | 0.95915 |
| without crosses | 0.95845 |
| without market context | 0.95816 |
| without user ratios | 0.95787 |

Social ratings give the largest gain. Market context and user ratios are neutral: removing both
gives 0.95820 vs 0.95812 averaged over 3 seeds, which is within seed noise.

## Engineering notes

- **Target leakage.** Leave-one-out averages leaked the label: validation RMSE got *worse* with
  every boosting round (0.971 → 0.981). Out-of-fold statistics fixed it.
- **Filter before parsing.** In user.json (2 GB), the user id sits at a fixed character offset. Lines
  are filtered on that id before `json.loads`, which skips more than 99% of the parsing.
- **Broadcast join.** Friends' ratings were first computed with a shuffle join, which took 66 s.
  A broadcast (map-side) lookup removed that stage, and the total time dropped from 209 s to 133 s
  with the same RMSE.
- **1 GB driver.** Features are packed per partition into float32 numpy blocks and pulled back with
  `toLocalIterator`, so the driver never holds Python lists of the whole matrix.
- **Reproducibility.** Folds come from a sort order plus row index rather than Python's per-process
  salted `hash()`. Model seeds are fixed.
- **Feature cache.** The cache is keyed by input sizes, mtimes and the pipeline version. With a warm
  cache, the transform stage takes 0.5 s instead of 16.2 s, and the total drops from 133 s to 118 s.
- **What did not help:** item-based CF and matrix factorization (ALS) as features, tried in earlier
  versions. Item-CF cost about 50 s for an RMSE change of about 0.001.
