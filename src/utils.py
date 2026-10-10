"""
utils.py - pipeline infrastructure and small pure helpers.

    RunReport      per-stage timing and metrics (observability)
    FeatureStore   on-disk cache of the entity tables, invalidated by an input fingerprint
    helpers        fault-tolerant JSON parsing, vector sums, text / geo utilities
"""
import json
import math
import os
import re
import shutil
import time

from config import NAN, PIPELINE_VERSION, WORD


# ============================================================================ observability
class RunReport(object):
    """Collects per-stage wall time and named metrics (counts, data-quality findings, scores)
    for one pipeline run, and renders them as a plain-text report at the end."""

    def __init__(self):
        self.stages = []      # (stage name, seconds)
        self.metrics = []     # (stage name, metric name, value)
        self._start = time.time()

    def stage(self, name):
        """Context manager timing one stage: `with report.stage('extract'): ...`."""
        report = self

        class _Timer(object):
            def __enter__(self):
                self.t = time.time()
                return self

            def __exit__(self, *exc):
                report.stages.append((name, time.time() - self.t))

        return _Timer()

    def metric(self, stage, name, value):
        """Record one named value under a stage."""
        self.metrics.append((stage, name, value))

    def total(self):
        """Seconds since the report was created (i.e. total pipeline time)."""
        return time.time() - self._start

    def render(self):
        """Stage timings followed by all metrics, as text."""
        out = ['', '=== Pipeline run report ===', '%-12s %8s' % ('stage', 'seconds')]
        out += ['%-12s %8.1f' % (n, s) for n, s in self.stages]
        out.append('%-12s %8.1f' % ('TOTAL', self.total()))
        out.append('--- metrics ---')
        out += ['%-10s %-38s %s' % m for m in self.metrics]
        return '\n'.join(out)


# ============================================================================ feature cache
class FeatureStore(object):
    """Tiny on-disk feature cache. Each table is saved as an RDD pickle file next to a manifest
    holding the input fingerprint (file sizes + mtimes + PIPELINE_VERSION); a table is rebuilt only
    when the fingerprint changed. Disabled when no cache directory is given.

    Measured: with a warm cache the transform stage takes 0.5s instead of 16.2s (total 133s -> 118s)."""

    def __init__(self, sc, root, inputs):
        self.sc, self.root = sc, root
        self.fingerprint = None
        if root:
            # Cheap fingerprint: size + modification time of every input, not a content hash
            # (hashing ~3GB would cost more than the cache saves).
            parts = [PIPELINE_VERSION]
            for p in sorted(inputs):
                st = os.stat(p)
                parts.append('%s:%d:%d' % (os.path.basename(p), st.st_size, int(st.st_mtime)))
            self.fingerprint = '|'.join(parts)

    def load_or_build(self, name, build):
        """Return (table, status) - from the cache when the fingerprint matches, else build() it."""
        if not self.root:
            return build(), 'computed (no cache)'
        path = os.path.join(self.root, name)
        manifest = path + '.manifest'
        if os.path.exists(manifest) and open(manifest).read() == self.fingerprint:
            return self.sc.pickleFile(path).collect()[0], 'cache hit'
        value = build()
        if os.path.exists(path):
            shutil.rmtree(path)       # saveAsPickleFile refuses to overwrite
        self.sc.parallelize([value], 1).saveAsPickleFile(path)
        # The manifest is written last, so a run that dies mid-save leaves no valid cache entry.
        with open(manifest, 'w') as f:
            f.write(self.fingerprint)
        return value, 'computed and cached'


# ============================================================================ small helpers
def safe_json(line, bad):
    """json.loads that counts malformed lines in a Spark accumulator instead of failing the job."""
    try:
        return json.loads(line)
    except ValueError:
        bad.add(1)
        return None


def add_vec(a, b):
    """Element-wise sum of two equal-length vectors; the reduce function for all additive stats."""
    return [x + y for x, y in zip(a, b)]


def ratio(a, b):
    """a / b, or NaN when the denominator is zero."""
    return a / float(b) if b else NAN


def clean_attr(v):
    """Normalise attribute strings such as "u'average'" or "'free'" to "average" / "free"."""
    v = v.strip()
    if v.startswith("u'") or v.startswith('u"'):
        v = v[1:]
    return v.strip("'\"")


def split_cats(d):
    """Category names of a business record: "Pizza, Restaurants" -> ['Pizza', 'Restaurants']."""
    return [c.strip() for c in (d.get('categories') or '').split(',') if c.strip()]


def chain_key(d):
    """Normalised business name used to detect chains ("McDonald's" == "mcdonalds")."""
    return re.sub(r"[^a-z0-9]", '', (d.get('name') or '').lower())


def tokens(text):
    """Set of lower-cased words plus adjacent-word bigrams ("not_good") of a text.
    Bigrams let the lexicon tell "good" (+) from "not_good" (-)."""
    w = WORD.findall(text.lower())
    return set(w + [a + '_' + b for a, b in zip(w, w[1:])])


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two (lat, lon) points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 12742.0 * math.asin(min(1.0, math.sqrt(a)))   # 12742 = Earth's diameter in km
