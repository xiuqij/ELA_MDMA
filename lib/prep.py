"""
prep.py - data loading, cleaning, and flexible subsetting.

This module is deliberately foundational and low-level: it knows how to get from raw QC_table
CSVs to a clean, analysis-ready dataframe, and how to slice that dataframe by any combination
of day / phase / time_point / sex / age. It does NOT know about domains or statistics -
see domain_scores.py and stats_utils.py for those.

Typical usage:
    from prep import load_data, filter_data

    df = load_data(female_path='female_12h_filtered.csv', male_path='male_12h_filtered.csv')
    baseline_active = filter_data(df, phase='active', time_point='baseline')
    baseline_day1_only = filter_data(df, phase='active', time_point='baseline', days=[1])
"""
import numpy as np
import pandas as pd

INFO_COLS = ['day', 'phase', 'box', 'box_ID', 'mouse', 'time_bin', 'time_window', 'time_point',
             'exp', 'sex', 'age', 'mouse_ID', 'background', 'treatment', 'condition']
QC_COLS = ['qc_exclude', 'qc_speed_outlier', 'qc_exclude_timebin']

# ---------------------------------------------------------------------------
# Known data-quality issues, flagged by data owner (not visible as NaN/QC flags).
# Kept as a list of filter-dicts so new issues discovered later can just be appended here
# rather than requiring new code. Each entry's key/value pairs are ANDed together to define
# the rows to drop; only entries whose keys actually exist in the loaded df are applied
# (so a 12h-only issue is silently skipped when loading a finer-resolution file, etc.)
# ---------------------------------------------------------------------------
KNOWN_BAD_BLOCKS = [
    {
        # male_P35, day 1, active phase, MDMA (post-injection) session: only ~7 of 12h were
        # recorded (hours 0-5 of the active phase missing) for ALL mice in this block. Raw
        # duration/count-based totals for this specific block are deflated and not comparable
        # to the other (full-window) blocks.
        'sex': 'male', 'age': 'P35', 'day': 1, 'phase': 'active', 'time_point': 'MDMA',
        'note': 'male P35 day1 active MDMA session: hours 0-5 of active phase not recorded '
                '(confirmed via 1h-resolution file - rows are entirely absent, not NaN). '
                'See matched_window_recovery.py for how to recover a partial-window comparison '
                'instead of dropping outright.',
    },
]

# Columns representing "amount of an event/state" where NaN genuinely means "no such event was
# observed" -> 0 is the correct fill.
# NOT filled: *_mean_duration (mean of zero events is undefined, not 0), *_ratio (chasing/chased
# and *_event_ratio: 0/0 undefined direction, not 0), normDS / rank (no chase interactions ->
# hierarchy undefined, not 0).
NA_TO_ZERO_SUFFIXES = ('_count', '_duration', '_duration_fraction', '_event_rate',
                        '_frames', '_together_fraction', '_alone_fraction', '_fraction',
                        '_sum', '_co_occupancy')
# '_ratio' (not bare 'ratio'): 'duration' itself contains 'ratio', so the bare pattern silently
# excluded every *_duration / *_duration_fraction column from the zero-fill above.
NA_KEEP_PATTERNS = ('mean_duration', '_ratio', 'normDS', 'rank')


def drop_known_bad_blocks(df, verbose=True):
    """Drop rows matching any entry in KNOWN_BAD_BLOCKS. Add new entries to that list as new
    data-quality issues are discovered - no code changes needed here."""
    df = df.copy()
    for block in KNOWN_BAD_BLOCKS:
        keys = {k: v for k, v in block.items() if k != 'note' and k in df.columns}
        if not keys:
            continue
        mask = pd.Series(True, index=df.index)
        for k, v in keys.items():
            mask &= (df[k] == v)
        if mask.any() and verbose:
            print(f"[drop_known_bad_blocks] dropping {mask.sum()} rows: {block.get('note', keys)}")
        df = df[~mask]
    return df


def fill_no_event_nans(df):
    """NaN -> 0 for columns where NaN means 'no event observed'. Leaves mean_duration,
    ratio, normDS, rank untouched (see NA_KEEP_PATTERNS docstring above)."""
    df = df.copy()
    for col in df.columns:
        if any(p in col for p in NA_KEEP_PATTERNS):
            continue
        if col.endswith(NA_TO_ZERO_SUFFIXES) and df[col].dtype.kind in 'fi':
            df[col] = df[col].fillna(0.0)
    return df


def clean_infinite(df):
    """Replace +/-inf with NaN. A handful of rate/fraction columns (e.g. non_wall_event_rate,
    ramp2_duration_fraction - worst at 1h resolution, a few persist at 2h/3h) are effectively
    x/~0 when a mouse's observed time in a bin is ~zero, producing literal inf rather than NaN.
    Left as inf, these poison compute_domain_scores()'s z-scoring: mean/std of the reference
    group become NaN, silently NaN-ing that domain for EVERY mouse in the group, not just the
    offending row. Treated as missing (like ratio/normDS/rank), not auto-zeroed the way the
    *_duration_fraction/*_event_rate NaN convention would suggest, since 'no observed time to
    measure from' isn't the same as 'zero events'.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    return df


def bin_seconds(df):
    """Bin length in seconds, derived per row as nest_duration + outside_nest_duration. That sum
    is exactly the bin length at every resolution and in both phases (12h/6h/4h/3h/2h/1h/30min/
    15min all verified), so no resolution argument is needed - the table describes its own bin
    width."""
    return df['nest_duration'] + df['outside_nest_duration']


def renormalize_rates(df, normalize_by='out_of_nest'):
    """Choose the denominator for the *_duration_fraction / *_event_rate features (27 columns).

    'out_of_nest' (default) is a NO-OP: the published columns already are
    x / outside_nest_duration, verified exact against their own numerators at all 8 resolutions
    (0 of 27 columns mismatch), so nothing is recomputed and the loaded table is bit-identical to
    what it was before this function existed.

    'bin' rebuilds them against the BIN LENGTH instead (<f>_duration / bin_seconds,
    <f>_count / bin_seconds). Two reasons this matters at fine resolution:

      - Stability. outside_nest_duration is bin_length - nest_duration, so it collapses toward
        zero as bins shrink (1st percentile: 5896s at 12h, 76s at 2h, 0s at 30min) and the ratio
        explodes - exploration_score's max |z| reaches 861 at 2h and 1103 at 30min. Dividing by a
        constant removes that failure mode outright, with no rows dropped (max |z| -> 4.8 and 5.4).
      - Interpretation. outside_nest_duration is ITSELF a significant ELA-vs-CTRL difference (ELA
        nests ~1700s more per 12h, dz=-0.615, p=0.0015), so the default denominator divides by an
        outcome that differs between the groups being compared, partially cancelling the effect
        being measured. The two conventions answer different questions: 'out_of_nest' = "when
        active, how is time allocated?", 'bin' = "over the whole window, how much happens?".

    Affects exactly 4 of the 12 core domains - exploration_score, feeding_drinking_amount_score,
    inactivity_score and speeding_score, the four built entirely from these features. The other 8
    are bit-identical either way. Knock-on effects in add_derived_features() below:
    feeder_prox_pref / water_prox_pref are ratios of two SAME-denominator fractions so it cancels
    and they are unaffected; total_chase_exposure is a sum of two fractions and IS rescaled.
    """
    if normalize_by == 'out_of_nest':
        return df
    if normalize_by != 'bin':
        raise ValueError(f"normalize_by must be 'out_of_nest' or 'bin', got {normalize_by!r}")

    df = df.copy()
    denom = bin_seconds(df)
    for col in [c for c in df.columns if c.endswith('_duration_fraction')]:
        numerator = col[:-len('_duration_fraction')] + '_duration'
        if numerator in df.columns:
            df[col] = df[numerator] / denom
    for col in [c for c in df.columns if c.endswith('_event_rate')]:
        numerator = col[:-len('_event_rate')] + '_count'
        if numerator in df.columns:
            df[col] = df[numerator] / denom
    return df


def add_derived_features(df):
    """Raw-feature-level derived columns used by multiple domain scores. Computed here (once,
    at load time) rather than inside domain_scores.py so they're available to any downstream
    analysis, not just the domain-score pipeline."""
    df = df.copy()
    df['nest_fraction'] = df['nest_duration'] / (df['nest_duration'] + df['outside_nest_duration'])
    # Resource proximity preference: PROX zones sit closer to the nest than DIST zones, so the
    # balance between them is a spatial-preference measure, not an amount measure. Using
    # duration_fraction (rather than raw duration) is equivalent here since the common
    # denominator (outside_nest_duration) cancels in the ratio.
    df['feeder_prox_pref'] = df['feeder_prox_duration_fraction'] / \
        (df['feeder_prox_duration_fraction'] + df['feeder_dist_duration_fraction'])
    df['water_prox_pref'] = df['water_prox_duration_fraction'] / \
        (df['water_prox_duration_fraction'] + df['water_dist_duration_fraction'])
    # Total time engaged in a chase interaction, in EITHER role. This is a literal sum of raw
    # fractions (computed before any z-scoring) - see domain_scores.py docstring for why this
    # matters: chasing_duration_fraction and chased_duration_fraction can move in opposite
    # directions under a group contrast, so this must be summed first, not z-scored separately
    # and averaged (that would answer a different, more confusing question).
    df['total_chase_exposure'] = df['chasing_duration_fraction'] + df['chased_duration_fraction']
    return df


def load_data(female_path, male_path, drop_bad_blocks=True, fill_nans=True,
               normalize_by='out_of_nest', verbose=True):
    """Load + clean one resolution's worth of data (female + male files concatenated).
    Works for any resolution file (15min/30min/1h/2h/3h/4h/6h/12h) with the same column schema,
    EXCEPT normDS/rank which only exist at 12h/session resolution.

    normalize_by: denominator for the *_duration_fraction / *_event_rate features - see
    renormalize_rates(). 'out_of_nest' (default) leaves them exactly as published, so every
    existing caller is unaffected. Pass 'bin' at fine resolution (2h and below), where the
    out-of-nest denominator becomes unstable.
    """
    f = pd.read_csv(female_path)
    m = pd.read_csv(male_path)
    df = pd.concat([f, m], ignore_index=True)
    df = df[~df['qc_exclude'] & ~df['qc_exclude_timebin']].copy()
    if drop_bad_blocks:
        df = drop_known_bad_blocks(df, verbose=verbose)
    if fill_nans:
        df = fill_no_event_nans(df)
    df = clean_infinite(df)
    # Before add_derived_features, so the derived columns are built from whichever
    # normalization the caller asked for rather than a mix of the two.
    df = renormalize_rates(df, normalize_by=normalize_by)
    df = add_derived_features(df)
    return df


def filter_data(df, phase=None, time_point=None, days=None, sex=None, age=None,
                 background=None, treatment=None):
    """Flexible subsetting. Any argument left as None is not filtered on.
    `days` accepts a single int or a list/tuple of ints, e.g. days=1 or days=[1,2].

    Examples:
        filter_data(df, phase='active', time_point='baseline')
        filter_data(df, phase='active', time_point='baseline', days=[1])       # day-1 only
        filter_data(df, phase='active', time_point='MDMA', sex='male', age='P35')
    """
    out = df
    if phase is not None:
        out = out[out['phase'] == phase]
    if time_point is not None:
        out = out[out['time_point'] == time_point]
    if days is not None:
        days = [days] if np.isscalar(days) else list(days)
        out = out[out['day'].isin(days)]
    if sex is not None:
        out = out[out['sex'] == sex]
    if age is not None:
        out = out[out['age'] == age]
    if background is not None:
        out = out[out['background'] == background]
    if treatment is not None:
        out = out[out['treatment'] == treatment]
    return out.copy()
