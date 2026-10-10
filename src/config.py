"""
config.py - every tunable constant of the pipeline in one place.

Feature-block lengths (N_*) live here too: when an entity is missing (e.g. a test business that is
not in business.json) its block is filled with that many NaNs, so every row has the same width.
"""
import re

# ============================================================================ pipeline
PIPELINE_VERSION = 'v3.1'   # part of the feature-cache fingerprint; bump when feature logic changes
K_FOLDS = 5                 # folds for out-of-fold rating statistics
NAN = float('nan')          # XGBoost treats NaN as missing and learns a default branch for it

# ============================================================================ feature engineering
N_TOP_CATS = 60             # most frequent categories one-hot encoded
LEX_ALPHA = 30.0            # lexicon smoothing: score = sum(rating - mean) / (count + alpha)
LEX_MIN_COUNT = 20          # tokens seen in fewer reviews are ignored
RECENT_YEAR = 2016          # reviews from this year on count as "recent" for trend features
STARS_PRIOR = 10.0          # pseudo-reviews pulling a business's stars toward its category mean
WORD = re.compile(r"[a-z']+")   # tokenizer: lower-case words, apostrophes kept ("don't")

# Ordinal encodings for string-valued business attributes.
ATTR_MAP = {
    'NoiseLevel': {'quiet': 1, 'average': 2, 'loud': 3, 'very_loud': 4},
    'RestaurantsAttire': {'casual': 1, 'dressy': 2, 'formal': 3},
    'Alcohol': {'none': 0, 'beer_and_wine': 1, 'full_bar': 2},
    'WiFi': {'no': 0, 'free': 1, 'paid': 2},
    'Smoking': {'no': 0, 'outdoor': 1, 'yes': 2},
    'BYOBCorkage': {'no': 0, 'yes_free': 1, 'yes_corkage': 2},
    'AgesAllowed': {'allages': 0, '18plus': 1, '19plus': 2, '21plus': 3},
}
# 'True' / 'False' attributes -> 1 / 0.
BOOL_ATTRS = ['BusinessAcceptsCreditCards', 'BikeParking', 'GoodForKids', 'RestaurantsTakeOut',
              'OutdoorSeating', 'RestaurantsGoodForGroups', 'WheelchairAccessible',
              'RestaurantsDelivery', 'RestaurantsReservations', 'HasTV', 'ByAppointmentOnly',
              'RestaurantsTableService', 'Caters', 'DogsAllowed', 'BusinessAcceptsBitcoin',
              'AcceptsInsurance', 'HappyHour', 'GoodForDancing', 'CoatCheck', 'DriveThru']
# Attributes stored as stringified dicts ("{'garage': False, 'lot': True}") -> number of Trues.
DICT_ATTRS = ['BusinessParking', 'Ambience', 'GoodForMeal', 'Music', 'BestNights']
PHOTO_LABELS = ['food', 'inside', 'outside', 'drink', 'menu']
COMPLIMENTS = ['hot', 'more', 'profile', 'cute', 'list', 'note', 'plain', 'cool', 'funny',
               'writer', 'photos']

# Lengths of the fixed-size feature blocks (used to fill missing entities with NaN).
N_USER_F = 15 + len(COMPLIMENTS)   # features.user_features()
N_PHOTO = 1 + len(PHOTO_LABELS)    # photo counts
N_CHECKIN = 6                      # features.checkin_profile()
N_CTX = 11                         # features.market_context()
N_STAT = 13                        # features.review_vector() / agg_stats()
N_SOCIAL = 4                       # features.social_features()

# ============================================================================ model
# Three XGBoost configurations whose predictions are averaged: deep/slow, medium, and
# shallow/strongly regularised. Their errors differ, so the average is steadier than any one.
# Only parameter names understood by both xgboost 0.72 (grading server) and current releases are used.
MODELS = [
    ({'max_depth': 10, 'eta': 0.02, 'min_child_weight': 200, 'colsample_bytree': 0.4,
      'subsample': 0.7, 'lambda': 10.0}, 900),
    ({'max_depth': 8, 'eta': 0.03, 'min_child_weight': 100, 'colsample_bytree': 0.5,
      'subsample': 0.8, 'lambda': 10.0}, 700),
    ({'max_depth': 6, 'eta': 0.04, 'min_child_weight': 50, 'colsample_bytree': 0.6,
      'subsample': 0.8, 'lambda': 20.0, 'alpha': 2.0}, 1100),
]
