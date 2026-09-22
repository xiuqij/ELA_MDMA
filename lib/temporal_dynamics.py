"""
temporal_dynamics.py - time-course of a raw feature or domain score across the active (or
inactive) phase, at any resolution (1h/2h/3h/4h/6h/12h), split by background (CTRL/ELA) or by
condition (CTRL_saline/CTRL_MDMA/ELA_saline/ELA_MDMA).

Generalizes two throwaway exploratory scripts (1_analysis_v1/fig7_temporal_day1.py and
1_analysis_v1/fig9_temporal_domains.py) into one reusable compute + plot pair built on top of
prep.py / aggregate.py / domain_scores.py, so it slots into the run_analysis.py flow the same way
treatment_interaction.py does. Differences from those two scripts (deliberate, not oversights):

  - No P35/P42 split. Ages are POOLED per mouse_ID (i.e. treated as repeated observations of the
    same mouse, not a separate factor) - drop 'age' from a custom id_cols list to restore a split
    if a later question needs it. See MOUSE_ID_COLS in run_analysis.py, which DOES include 'age'
    (each mouse_ID is tested once at P35 and once at P42) - TEMPORAL_ID_COLS here intentionally
    omits it.
  - Works on domain scores AND raw features with the same call: aggregate_to_mouse() keeps raw
    feature columns, and compute_domain_scores() only ADDS composite columns on top - so any
    column name, raw or composite, can be handed to plot_temporal()/plot_temporal_grid().
  - group='condition' uses the `condition` column already present in the QC_table
    (background + '_' + treatment, e.g. 'ELA_MDMA') to give a 4-way split, not just the 2-way
    background split - this is what makes the module usable for step 7 (both time_points /
    treatments in play), not just step 6 (baseline-only).
  - Day selection/collapse works the same way run_analysis.py step 1 handles it for the 12h
    tables (baseline_active / _day1 / _day23): load_temporal_data(days=...) picks which day(s)
    to include, and keep_day=False (default) AVERAGES them into one within-day shape per window -
    e.g. days=[2, 3] gives a day2+3-average trajectory, matching mouse_baseline_day23. Pass
    keep_day=True instead to keep 'day' as its own axis and see the actual multi-day trajectory
    (one point per day per window, not collapsed) - plot_temporal() then switches to a continuous
    (day, window) x-axis automatically.

Two intended call sites (see run_analysis.py steps 6 & 7):
  - Step 6 (baseline only): group='background', time_points='baseline' -> 2 lines/panel, the
    within-day ELA-vs-CTRL shape at baseline.
  - Step 7 (post-injection, ABSOLUTE-value view - a companion to treatment_interaction.py's DELTA
    time-course, not a replacement): group='condition', time_points='MDMA' -> 4 lines/panel, the
    within-session shape for each of the 4 background x treatment groups.

Z-scoring reference: domain scores are z-scored against `reference_time_point` rows (default
'baseline'), POOLED ACROSS ALL WINDOWS per sex - deliberately different from
treatment_interaction.compute_window_deltas() (which references each window against ITS OWN
same-window baseline, to isolate a treatment delta from the daily rhythm). Here the within-day
trend IS the thing being plotted, so a pooled reference keeps it visible instead of flattening it.

A third view, plot_session_trajectory()/plot_session_trajectory_grid(), puts BOTH sessions in one
panel: baseline days 1-3, a visual break at the treatment boundary, then post-injection days 1-3
- the full arc in one picture rather than two separate plots. Load with time_points=('baseline',
'MDMA') and keep_day=True for this. Pass baseline_group='background' to collapse just the
baseline block to the 2-way CTRL/ELA split (no treatment has happened yet, so splitting it by
`group` - typically 'condition', 4-way - there too is not meaningful) while the post-injection
block keeps the full split.

Both grid functions (plot_temporal_grid, plot_session_trajectory_grid) wrap long feature lists
into extra row-groups automatically (or via `max_cols`) instead of one very wide row, accept
pair_sexes=True to place each feature's male/female panels side by side on a shared y-axis scale
instead of stacking sexes in separate rows, and draw a single figure-level legend_loc='top'/
'bottom' legend instead of a per-panel one - same pattern as plot_domain_score_by_day.py's
plot_domain_scores_by_day_grid().

Usage:
    from temporal_dynamics import (
        load_temporal_data, plot_temporal, plot_temporal_grid,
        plot_session_trajectory, plot_session_trajectory_grid,
    )
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from prep import load_data, filter_data
from aggregate import aggregate_to_mouse
from domain_scores import compute_domain_scores, CORE_DOMAINS

# mouse_ID alone isn't unique across ages (each mouse is tested at P35 AND P42) - but 'age' is
# deliberately left out here so the two ages get pooled together per window/time_point (see
# module docstring). box_ID is kept for completeness even though it's not used for pairing here.
TEMPORAL_ID_COLS = ['mouse_ID', 'sex', 'background', 'treatment', 'condition', 'box_ID']

# Palette conventions consistent with treatment_interaction.py / plot_effect_table.py.
COLOR_CTRL = '#898781'
COLOR_MUTED = '#898781'
COLOR_INK = '#0b0b0b'
SEX_ELA_COLORS = {'female': '#e34948', 'male': '#2a78d6'}

# Styling for overlaying multiple time_points (sessions) in one panel (reproduces
# fig7_temporal_day1.py's baseline=dashed/square vs MDMA=solid/circle convention). Only used
# when the data handed to plot_temporal() contains more than one time_point AND group=
# 'background' (2 colors) - group='condition' uses TREATMENT_STYLE instead (see below).
TIME_POINT_STYLE = {
    'baseline': dict(marker='s', ls='--', alpha=0.6),
    'MDMA': dict(marker='o', ls='-', alpha=0.95),
}

# Styling for the 4-way group='condition' split: color already encodes background (see
# _condition_palette below - CTRL_saline/CTRL_MDMA share a color, ELA_saline/ELA_MDMA share the
# other), so treatment (saline vs MDMA, the box-level dosing group baked into `condition`) has to
# be encoded some OTHER way or CTRL_saline and CTRL_MDMA would be visually identical. Marker/
# linestyle/alpha here do that job - alpha convention (saline fainter) matches treatment_
# interaction.py's plot_deltas_by_condition_grid.
TREATMENT_STYLE = {
    'saline': dict(marker='o', ls='--', alpha=0.45),
    'MDMA': dict(marker='s', ls='-', alpha=0.95),
}

# Styling for a baseline session drawn as a REFERENCE UNDERLAY beneath the post-treatment lines
# (plot_temporal(..., baseline_as_reference=True)) rather than as an equal-status second series.
# Marker-less dotted, so it reads as "where this group started" without competing with the
# post-treatment line's own marker/linestyle encoding (which is already carrying treatment when
# group='condition') - same solid-is-tested / dotted-is-reference grammar as
# treatment_interaction.plot_delta_by_window's saline reference line.
BASELINE_REFERENCE_STYLE = dict(marker='', ls=':', alpha=0.8, lw=1.4)

# Default per-BLOCK captions for plot_session_trajectory()'s two side-by-side session blocks.
# The caller can override either one (session_labels=) to state what the block actually contains -
# e.g. 'baseline (days 2-3 avg)' when the baseline side was averaged over days 2-3, which the
# function cannot infer because that averaging happens upstream in
# compute_posttx_window_mouse_sess() and leaves no day column behind.
SESSION_BLOCK_LABELS = {'baseline': 'baseline', 'MDMA': 'after treatment'}

GROUP_ORDERS = {
    'background': ['CTRL', 'ELA'],
    'condition': ['CTRL_saline', 'CTRL_MDMA', 'ELA_saline', 'ELA_MDMA'],
}


def _background_palette(sex):
    return {'CTRL': COLOR_CTRL, 'ELA': SEX_ELA_COLORS.get(sex, '#e34948')}


def _condition_palette(sex):
    ela = SEX_ELA_COLORS.get(sex, '#e34948')
    return {'CTRL_saline': COLOR_CTRL, 'CTRL_MDMA': COLOR_CTRL,
            'ELA_saline': ela, 'ELA_MDMA': ela}


GROUP_PALETTES = {'background': _background_palette, 'condition': _condition_palette}


def _domain_label(feature):
    return feature.replace('_score', '').replace('_', ' ').capitalize()


def _is_sex_variant_group(grp):
    """True if `grp`'s color depends on sex - the 'ELA' side of GROUP_PALETTES (background=
    'ELA', condition='ELA_saline'/'ELA_MDMA'), which is colored via SEX_ELA_COLORS; the 'CTRL'
    side is COLOR_CTRL regardless of sex. Used to decide which legend entries need a sex
    suffix when a grid's shared legend (_shared_legend_handles) mixes both sexes - without it,
    two differently-colored lines both labeled plain 'ELA' collide on that label and the dedup
    silently drops one sex's handle (see plot_temporal_grid/plot_session_trajectory_grid)."""
    return grp.split('_', 1)[0] == 'ELA'


def _auto_max_cols(n_items, small_threshold, min_cols):
    """Pick a column count that keeps a wrapped grid roughly square once n_items exceeds
    small_threshold, instead of always laying everything out in one long row (same helper
    as plot_domain_score_by_day.py's grid wrapping)."""
    if n_items <= small_threshold:
        return n_items
    return max(min_cols, int(np.ceil(np.sqrt(n_items))))


def _shared_legend_handles(axes_flat):
    """Collect one handle per distinct label across all panels in a grid, in first-seen
    order - labelled artists exist on an axes regardless of whether ax.legend() was ever
    called on it, so panels can be plotted with show_legend=False and still contribute
    here. Used to build a single figure-level legend instead of a per-panel one."""
    seen = {}
    for ax in axes_flat:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            seen.setdefault(ll, hh)
    return list(seen.values()), list(seen.keys())


def _legend_layout(legend_loc, title):
    """tight_layout padding + suptitle/legend y-positions for a shared figure-level legend
    placed 'top' or 'bottom' relative to the grid (mirrors plot_domain_score_by_day.py's
    pad scheme, kept in sync so grids from either module stack/compose predictably)."""
    top_pad, bottom_pad, title_y, legend_y = 0.0, 0.0, 1.02, None
    if legend_loc == 'top':
        top_pad, legend_y = (0.12, 0.935) if title else (0.07, 0.965)
        title_y = 0.99
    elif legend_loc == 'bottom':
        bottom_pad, legend_y = 0.07, 0.015
        if title:
            top_pad, title_y = 0.06, 0.99
    return top_pad, bottom_pad, title_y, legend_y


def _draw_shared_legend(fig, axes_flat, legend_loc, legend_y):
    if not legend_loc:
        return
    handles, labels = _shared_legend_handles(axes_flat)
    if handles:
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, legend_y),
                   ncol=min(len(handles), 4), frameon=False, fontsize=8.5)


def load_temporal_data(resolution, domain_defs=None, extra_features=None, id_cols=None,
                        phase='active', time_points=None, days=None, keep_day=False,
                        reference_time_point='baseline', data_path=None, female_path=None,
                        male_path=None, normalize_by='out_of_nest', verbose=True):
    """Load one resolution's QC_table (1h/2h/3h/4h/6h/12h - same schema, pass just the number+h,
    e.g. '3h'), subset to `phase` (and optionally `time_points` - a single value or list), and
    return one row per mouse per time_window (x time_point, if more than one is loaded) with both
    the raw features and z-scored domain-score columns (domain_defs, default CORE_DOMAINS) added.

    extra_features: any additional raw columns to carry through/plot that aren't already pulled
    in by domain_defs (e.g. a feature you want to look at on its own, not as part of a domain).

    days: which day(s) to include - None (all 3, the default), a single int (e.g. 1), or a list
    (e.g. [2, 3]). Passed straight through to prep.filter_data(days=...) - same day-selection
    pattern used for the 12h/session tables in run_analysis.py step 1 (baseline_active_day1 =
    days=1, baseline_active_day23 = days=[2, 3]).

    keep_day: if False (default), the selected day(s) are AVERAGED per mouse/window/time_point -
    e.g. days=[2, 3], keep_day=False gives one day2+3-averaged row per mouse per window (same
    idea as run_analysis.py's mouse_baseline_day23 - one collapsed within-day shape). If True,
    'day' is kept as its own grouping column instead of being averaged away, giving one row per
    mouse per DAY per window per time_point - pass this to plot_temporal()/plot_temporal_grid()
    to see an actual continuous multi-day trajectory (day-block-aware x-axis) rather than a
    single day(s)-collapsed shape.

    normalize_by: passed straight to prep.load_data() - 'out_of_nest' (default, unchanged) or
    'bin'. Exposed here so a fine-resolution call through this function gets the same denominator
    treatment as one that calls load_data() directly, instead of silently keeping the unstable
    out-of-nest denominator (see prep.renormalize_rates()).

    normDS/rank (used by social_hierarchy_score) don't exist below 12h resolution - dropped
    automatically, same as treatment_interaction.compute_window_deltas().

    If `reference_time_point` isn't among the loaded time_points (e.g. you only loaded
    time_points='MDMA'), falls back to self-referencing (z-scored against the loaded session's
    own distribution) with a warning - pass both time_points if you want a baseline-referenced
    scale.
    """
    domain_defs = domain_defs or CORE_DOMAINS
    id_cols = id_cols or TEMPORAL_ID_COLS
    data_path = data_path or ''
    female_path = female_path or os.path.join(data_path, f'female_{resolution}_filtered.csv')
    male_path = male_path or os.path.join(data_path, f'male_{resolution}_filtered.csv')

    df = load_data(female_path=female_path, male_path=male_path,
                    normalize_by=normalize_by, verbose=verbose)
    sub = filter_data(df, phase=phase, days=days)
    if time_points is not None:
        time_points = [time_points] if isinstance(time_points, str) else list(time_points)
        sub = sub[sub['time_point'].isin(time_points)].copy()

    raw_features = sorted({f for spec in domain_defs.values()
                            for f in spec['features'] + spec.get('flip', [])}
                           | set(extra_features or []))
    missing = [f for f in raw_features if f not in sub.columns]
    if missing and verbose:
        print(f"[load_temporal_data] {missing} not available at {resolution} resolution, "
              f"dropped (domains using them are scored on their remaining features only)")
    raw_features = [f for f in raw_features if f in sub.columns]

    group_cols = list(id_cols) + ['time_point', 'time_window'] + (['day'] if keep_day else [])
    mouse_win = aggregate_to_mouse(sub, group_cols, raw_features)

    ref_mask = mouse_win['time_point'] == reference_time_point
    if not ref_mask.any():
        if verbose:
            print(f"[load_temporal_data] reference_time_point={reference_time_point!r} not in "
                  f"loaded data (time_points={time_points}) - self-referencing against all "
                  "loaded rows instead")
        ref_mask = pd.Series(True, index=mouse_win.index)
    mouse_win = compute_domain_scores(mouse_win, domain_defs=domain_defs, reference_mask=ref_mask)
    return mouse_win


def ordered_windows(data, window_col='time_window'):
    """Sort time_window labels ('12-18','18-24',... or '0-3','3-6',... or, at 15/30-min
    resolution, '12.0-12.5','12.5-13.0',...) by their numeric start hour. Valid within a single
    phase (windows returned by load_temporal_data don't wrap midnight within a phase - see
    prep.py). float(), not int(), since sub-hour resolutions have a fractional start hour
    ('12.0' etc.) - int('12.0') raises ValueError."""
    return sorted(data[window_col].unique().tolist(), key=lambda w: float(w.split('-')[0]))


def ordered_day_windows(data, day_col='day', window_col='time_window'):
    """Sort (day, time_window) combinations for a continuous multi-day x-axis: primarily by day,
    then by each window's numeric start hour within that day. Input: `data` with load_temporal_
    data(..., keep_day=True)'s 'day' column present. float() - see ordered_windows()."""
    combos = data[[day_col, window_col]].drop_duplicates().itertuples(index=False, name=None)
    return sorted(combos, key=lambda dw: (dw[0], float(dw[1].split('-')[0])))


def format_window_label(w, style='relative', active_start_hour=12, unit=None):
    """Format a time_window label ('12.0-12.5', '12-16', ...) for display.

    style='zt' - the raw label unchanged, prefixed 'ZT' (e.g. 'ZT12.0-12.5') - for
    troubleshooting/cross-referencing against the raw QC_table files.

    style='relative' (default) - time since `active_start_hour` (treatment/lights-off),
    auto-scaled to minutes when the window span is under 1h (e.g. '0-30min') or hours otherwise
    (e.g. '2-4h') - more intuitive for a general audience than a ZT clock value. Pass
    unit='min'/'h' to force one unit across a mixed-resolution grid instead of auto-scaling.
    """
    lo, hi = (float(x) for x in w.split('-'))
    if style == 'zt':
        return f'ZT{lo:g}-{hi:g}'
    rel_lo, rel_hi = lo - active_start_hour, hi - active_start_hour
    use_min = unit == 'min' or (unit is None and (hi - lo) < 1)
    if use_min:
        return f'{rel_lo * 60:g}-{rel_hi * 60:g}min'
    return f'{rel_lo:g}-{rel_hi:g}h'


def plot_temporal(data, feature, sex, ax, group='background', window_col='time_window',
                   day_col='day', show_legend=True, sex_suffix=None, value_suffix='',
                   show_baseline=True, baseline_as_reference=False, label_style=None,
                   active_start_hour=12):
    """One feature's (or domain score's) time-course across time_window, for one sex, split by
    `group` - 'background' (CTRL vs ELA, 2 lines) or 'condition' (the 4 background x treatment
    groups, 4 lines).

    value_suffix: appended to `feature` to build the column actually plotted, so a DELTA table
    (aggregate.mouse_level_deltas() output, whose columns are '<feature>__delta') can be handed
    to this same function with value_suffix='__delta' - panel titles still use the plain feature
    name. Same convention as plot_by_box.plot_domains_by_box_grid(value_suffix=...). Delta tables
    have no 'time_point' column; that's handled (a single unlabelled series is drawn).

    show_baseline: False drops the 'baseline' session entirely, leaving only the post-treatment
    lines - the same figure without the reference clutter, so a panel can be re-saved with or
    without it by flipping one flag rather than re-filtering the table. Ignored when `data` has
    no 'time_point' column (e.g. a delta table).

    baseline_as_reference: when `data` holds both sessions, draw the 'baseline' time_point as a
    marker-less dotted underlay (BASELINE_REFERENCE_STYLE) instead of as an equal-status second
    series, and drop the '(MDMA)' qualifier from the post-treatment labels. Use this for the
    "post-treatment value with baseline overlaid" view; leave False to give both sessions equal
    weight (the fig7-style baseline-vs-MDMA overlay, group='background'). Has no effect when
    show_baseline=False. For a version that puts the two sessions in SEPARATE side-by-side blocks
    instead of overlaying them - easier to read when the overlay gets busy - see
    plot_session_trajectory()/plot_session_trajectory_grid() below, which accept these same
    window tables.

    label_style: None (default) keeps the raw time_window labels ('12-16', '12.0-12.5'). Pass
    'relative' for audience-friendly time-since-`active_start_hour` labels ('0-30min', '2-4h') or
    'zt' for explicit 'ZT12-16' - see format_window_label(). 'relative' assumes the windows sit in
    the phase starting at `active_start_hour`, so leave it None/'zt' for inactive-phase windows.

    sex_suffix: if given (e.g. 'male'), appended as a qualifier to sex-VARIANT group labels
    only (the 'ELA' side - see _is_sex_variant_group) - e.g. 'ELA' -> 'ELA (male)'. CTRL stays
    plain since its color doesn't depend on sex. Pass this from a grid that plots more than one
    sex into a single shared legend (plot_temporal_grid), so the two sexes' differently-colored
    ELA lines don't collide on the same legend label and silently lose one entry to dedup - not
    needed for a single-sex panel/legend, where there's no such collision to begin with.

    group='condition' styles each line by its treatment half (saline vs MDMA, via TREATMENT_
    STYLE - saline fainter/dashed, MDMA solid) since color alone (background) can't tell
    CTRL_saline from CTRL_MDMA apart. group='background' instead styles by time_point (session)
    IF `data` contains more than one (via TIME_POINT_STYLE: baseline=dashed/square, MDMA=solid/
    circle) - this is what lets the same function reproduce fig7_temporal_day1.py's baseline-vs-
    MDMA overlay. If `data` has only one time_point (the common case - see load_temporal_data
    (time_points=...)), group='background' lines are plain solid.

    If `data` has a `day_col` with more than one distinct value (i.e. loaded with
    load_temporal_data(..., keep_day=True)), the x-axis becomes a continuous (day, time_window)
    sequence spanning all loaded days - light alternating shading + a 'Day N' label mark each
    day's block - so the actual trajectory across days is visible, not just a single collapsed
    within-day shape. Otherwise the x-axis is just time_window (the original single-cycle view).

    Input: `data` - output of load_temporal_data().
    Returns: ax
    """
    if group not in GROUP_ORDERS:
        raise ValueError(f"group must be one of {list(GROUP_ORDERS)}, got {group!r}")
    col = f'{feature}{value_suffix}'
    sub_sex = data[data.sex == sex]
    if not show_baseline and 'time_point' in sub_sex.columns:
        sub_sex = sub_sex[sub_sex['time_point'] != 'baseline']
    order = [g for g in GROUP_ORDERS[group] if g in sub_sex[group].unique()]
    palette = GROUP_PALETTES[group](sex)
    # Delta tables (value_suffix='__delta') carry no session column - one unlabelled series.
    has_time_point = 'time_point' in sub_sex.columns
    if has_time_point:
        time_points = [t for t in TIME_POINT_STYLE if t in sub_sex['time_point'].unique()] + \
            [t for t in sub_sex['time_point'].unique() if t not in TIME_POINT_STYLE]
    else:
        time_points = [None]

    multi_day = day_col in sub_sex.columns and sub_sex[day_col].nunique() > 1
    if multi_day:
        day_windows = ordered_day_windows(sub_sex, day_col, window_col)
        x = np.arange(len(day_windows))
        xticklabels = [w for _, w in day_windows]
    else:
        windows = ordered_windows(sub_sex, window_col)
        x = np.arange(len(windows))
        xticklabels = windows

    for grp in order:
        color = palette[grp]
        if group == 'condition':
            _, treat = grp.split('_', 1)
            cond_style = TREATMENT_STYLE.get(treat, dict(marker='o', ls='-', alpha=0.9))
        else:
            cond_style = None
        for tp in time_points:
            d = sub_sex[sub_sex[group] == grp]
            if has_time_point:
                d = d[d['time_point'] == tp]
            if multi_day:
                g = d.groupby([day_col, window_col])[col].agg(['mean', 'sem']).reindex(day_windows)
            else:
                g = d.groupby(window_col)[col].agg(['mean', 'sem']).reindex(windows)

            is_reference = baseline_as_reference and tp == 'baseline'
            if is_reference:
                style = BASELINE_REFERENCE_STYLE
                qualifiers = ['baseline']
            else:
                style = cond_style or TIME_POINT_STYLE.get(tp, dict(marker='o', ls='-', alpha=0.9))
                qualifiers = [tp] if (len(time_points) > 1 and not baseline_as_reference) else []
            if sex_suffix and _is_sex_variant_group(grp):
                qualifiers = qualifiers + [sex_suffix]
            label = f'{grp} ({", ".join(qualifiers)})' if qualifiers else grp

            kw = dict(capsize=3, lw=2, markersize=5)
            kw.update(style)
            # No error bars on a reference underlay - it would clutter the post-treatment series
            # it sits behind, which keeps its own mean+/-SEM.
            ax.errorbar(x, g['mean'], yerr=None if is_reference else g['sem'], color=color,
                        label=label, zorder=1 if is_reference else 2, **kw)

    if multi_day:
        days_seen = sorted({d for d, _ in day_windows})
        block_start = {d: next(i for i, (dd, _) in enumerate(day_windows) if dd == d)
                        for d in days_seen}
        boundaries = [block_start[d] for d in days_seen] + [len(day_windows)]
        for i, d in enumerate(days_seen):
            lo, hi = boundaries[i] - 0.5, boundaries[i + 1] - 0.5
            if i % 2 == 1:
                ax.axvspan(lo, hi, color=COLOR_MUTED, alpha=0.06, zorder=0)
            if i > 0:
                ax.axvline(lo, color=COLOR_MUTED, lw=0.6, alpha=0.4, zorder=0)
            ax.text((lo + hi) / 2, 0.97, f'Day {d}', transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=7, color=COLOR_MUTED)

    if label_style:
        xticklabels = [format_window_label(w, style=label_style,
                                            active_start_hour=active_start_hour)
                        for w in xticklabels]
        xlabel = ('Time since active-phase start' if label_style == 'relative'
                   else 'Zeitgeber time, ZT (h)')
    else:
        xlabel = 'Zeitgeber time, ZT (h)'

    ax.set_xticks(x)
    if multi_day:
        ax.set_xticklabels(xticklabels, fontsize=6.5, rotation=90)
    else:
        ax.set_xticklabels(xticklabels, fontsize=7.5, rotation=45, ha='right')
    ax.set_xlabel(xlabel + (', by day' if multi_day else ''), fontsize=8.5)
    ax.axhline(0, color=COLOR_MUTED, lw=0.5, ls=':', zorder=0)
    ax.set_title(_domain_label(feature), fontsize=10.5)
    if show_legend:
        ax.legend(fontsize=7, loc='best')
    sns.despine(ax=ax)
    return ax


def plot_temporal_grid(data, features, sexes=('female', 'male'), group='background',
                        window_col='time_window', day_col='day', pair_sexes=False,
                        max_cols=None, legend_loc='top', ylabel=None, value_suffix='',
                        show_baseline=True, baseline_as_reference=False, label_style=None,
                        active_start_hour=12, title=None, save_path=None, show=True, dpi=150):
    """Reference grid of plot_temporal() panels, one per feature x sex.

    Two layouts are available:

    - pair_sexes=False (default): one row per sex, one column per feature (as before).
      When `features` is long, the columns wrap into additional row-groups (stacked
      vertically, len(sexes) rows per group) instead of producing one very wide row,
      with `max_cols` (or an automatic sqrt(n)-based choice) controlling the wrap width.
    - pair_sexes=True: requires exactly two sexes. Each feature gets its own male/female
      pair of panels placed side by side, y-axis matched across the pair so the two
      trajectories are directly comparable, with the second panel's y tick labels
      suppressed since it shares the first panel's scale. Feature-pairs wrap into a grid
      the same way when there are many features.

    A single figure-level legend (deduped across all panels, so it still shows every
    group/time_point/treatment style actually used) replaces the old per-panel corner
    legend - legend_loc='top'/'bottom' places it above/below the whole grid, or None to
    omit it.

    Input: `data` - output of load_temporal_data(). Panels widen automatically when `data`
    is multi-day (load_temporal_data(..., keep_day=True)) to fit the longer x-axis.

    Returns: (fig, axes)
    """
    multi_day = day_col in data.columns and data[day_col].nunique() > 1
    panel_w = 4.2 if multi_day else 2.8
    n_feat = len(features)

    if pair_sexes:
        if len(sexes) != 2:
            raise ValueError('pair_sexes=True requires exactly two sexes')
        sex_left, sex_right = sexes
        ncols_blocks = min(n_feat, max_cols or _auto_max_cols(n_feat, small_threshold=3, min_cols=2))
        nrows_blocks = int(np.ceil(n_feat / ncols_blocks))

        fig, axes = plt.subplots(nrows_blocks, ncols_blocks * 2,
                                  figsize=(panel_w * ncols_blocks * 2, 3.9 * nrows_blocks),
                                  squeeze=False)

        for i, feat in enumerate(features):
            row, col_block = divmod(i, ncols_blocks)
            ax_left, ax_right = axes[row, col_block * 2], axes[row, col_block * 2 + 1]
            panel_kw = dict(group=group, window_col=window_col, day_col=day_col,
                             show_legend=False, value_suffix=value_suffix,
                             show_baseline=show_baseline,
                             baseline_as_reference=baseline_as_reference,
                             label_style=label_style, active_start_hour=active_start_hour)
            plot_temporal(data, feat, sex_left, ax_left, sex_suffix=sex_left, **panel_kw)
            plot_temporal(data, feat, sex_right, ax_right, sex_suffix=sex_right, **panel_kw)

            ylo = min(ax_left.get_ylim()[0], ax_right.get_ylim()[0])
            yhi = max(ax_left.get_ylim()[1], ax_right.get_ylim()[1])
            ax_left.set_ylim(ylo, yhi)
            ax_right.set_ylim(ylo, yhi)
            ax_right.tick_params(labelleft=False)

            domain_label = _domain_label(feat)
            ax_left.set_title(f'{domain_label}\n{sex_left}', fontsize=8.5)
            ax_right.set_title(f'{domain_label}\n{sex_right}', fontsize=8.5)
            if col_block == 0:
                ax_left.set_ylabel(ylabel or 'z-score', fontsize=9)

        for i in range(n_feat, nrows_blocks * ncols_blocks):
            row, col_block = divmod(i, ncols_blocks)
            axes[row, col_block * 2].axis('off')
            axes[row, col_block * 2 + 1].axis('off')

    else:
        ncols = min(n_feat, max_cols or _auto_max_cols(n_feat, small_threshold=6, min_cols=3))
        n_groups = int(np.ceil(n_feat / ncols))
        nrows = n_groups * len(sexes)

        fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, 3.9 * nrows), squeeze=False)

        for grp_i in range(n_groups):
            feats_in_group = features[grp_i * ncols:(grp_i + 1) * ncols]
            for row_offset, sex in enumerate(sexes):
                r = grp_i * len(sexes) + row_offset
                for col, feat in enumerate(feats_in_group):
                    ax = axes[r, col]
                    plot_temporal(data, feat, sex, ax, group=group, window_col=window_col,
                                   day_col=day_col, show_legend=False,
                                   sex_suffix=sex if len(sexes) > 1 else None,
                                   value_suffix=value_suffix, show_baseline=show_baseline,
                                   baseline_as_reference=baseline_as_reference,
                                   label_style=label_style, active_start_hour=active_start_hour)
                    ax.set_title(_domain_label(feat) if row_offset == 0 else '', fontsize=9)
                    ax.set_ylabel(f'{sex}\n{ylabel}' if (col == 0 and ylabel) else
                                   (sex if col == 0 else ''), fontsize=9)
                for col in range(len(feats_in_group), ncols):
                    axes[r, col].axis('off')

    top_pad, bottom_pad, title_y, legend_y = _legend_layout(legend_loc, title)
    plt.tight_layout(rect=(0, bottom_pad, 1, 1 - top_pad))
    if title:
        fig.suptitle(title, color=COLOR_INK, fontsize=12, fontweight='bold', y=title_y)
    _draw_shared_legend(fig, axes.flatten(), legend_loc, legend_y)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes


def plot_session_trajectory(data, feature, sex, ax, group='condition', baseline_group=None,
                             session_order=('baseline', 'MDMA'), time_point_col='time_point',
                             day_col='day', window_col='time_window', gap=1.2, show_legend=True,
                             sex_suffix=None, value_suffix='', label_style=None,
                             active_start_hour=12, session_labels=None, caption_loc='below'):
    """One feature's (or domain score's) FULL trajectory across BOTH sessions in a single panel:
    baseline days 1-3 on the left, a visual break at the treatment boundary, then post-injection
    (MDMA) days 1-3 on the right - same within-day (day, time_window) x-axis logic as
    plot_temporal()'s multi-day mode, just placed in two side-by-side blocks (one per session)
    instead of overlaid on the same x-axis, since the two sessions are different points in time,
    not two readings of the same window. Each block's x-tick labels are the actual `time_window`
    values for its own days (e.g. '12-16', '16-20', ...), not a running bin count, so the same
    label always means the same time of day on both sides of the break.

    Input: `data` - output of load_temporal_data(..., time_points=session_order, keep_day=True).
    Z-scoring should be baseline-referenced (load_temporal_data's default) so 0 means "typical
    baseline" on BOTH sides of the break, making the jump at the treatment boundary meaningful.

    group='condition' (default) is the natural choice for the post-injection block(s) since both
    treatment arms are in play there - styled via TREATMENT_STYLE (saline fainter/dashed, MDMA
    solid), color by background, CONSTANT across the break so e.g. the ELA_MDMA line reads as the
    same group on both sides, just discontinuous. group='background' also works (2 lines, plain
    solid) if you don't need the treatment split anywhere.

    baseline_group: if given, overrides `group` for the FIRST session in `session_order` only
        (baseline, by convention) - e.g. baseline_group='background' collapses that block to the
        2-way CTRL/ELA split instead of the 4-way background x treatment split, since at baseline
        no drug has been given yet and a box's FUTURE treatment assignment splitting it into
        CTRL_saline/CTRL_MDMA (or ELA_saline/ELA_MDMA) there is not a meaningful distinction.
        Colors still line up across the break either way (both palettes color CTRL grey and ELA
        by sex - see GROUP_PALETTES); the legend simply gains extra entries for the baseline-only
        lines when `baseline_group` differs from `group`. None (default) keeps `group` uniform
        across the whole trajectory, matching the previous behavior.

    sex_suffix: see plot_temporal() - appended to sex-VARIANT group labels only ('ELA' side),
    for use from a grid sharing one legend across sexes (plot_session_trajectory_grid).

    value_suffix / label_style / active_start_hour: as in plot_temporal().

    session_labels: dict overriding the caption naming each session block (defaults to
    SESSION_BLOCK_LABELS - 'baseline' and 'after treatment'). Pass e.g.
    {'baseline': 'baseline (days 2-3 avg)'} to say what the baseline block actually averages,
    which this function can't work out for itself. The x-axis label keeps only the window
    description; which half is which is read off these captions instead.

    caption_loc: 'below' (default) writes each caption under its own block. 'title' or 'none'
    skip them here - 'title' is handled by plot_session_trajectory_grid(), which folds them into
    the panel title instead ('Exploration (baseline | after treatment)') for a more compact
    figure.

    Also accepts the single-window-block tables built by
    treatment_interaction.compute_posttx_window_mouse_sess() / compute_posttx_window_raw(), which
    have no `day_col`: a constant day is assumed and the per-day shading/labels are suppressed.
    That makes this the SEPARATE-PANELS alternative to plot_temporal(baseline_as_reference=True) -
    same data, but baseline and post-treatment sit in two side-by-side blocks split by the
    treatment break instead of being overlaid, which reads better when the overlay gets busy.

    Returns: ax
    """
    if day_col not in data.columns:
        data = data.assign(**{day_col: 1})
    col = f'{feature}{value_suffix}'
    for g in (group, baseline_group):
        if g is not None and g not in GROUP_ORDERS:
            raise ValueError(f"group must be one of {list(GROUP_ORDERS)}, got {g!r}")
    sub_sex = data[data.sex == sex]

    blocks = []  # (session, group_for_session, day_windows, xs)
    x0 = 0
    for i, sess in enumerate(session_order):
        sess_sub = sub_sex[sub_sex[time_point_col] == sess]
        if not len(sess_sub):
            continue
        sess_group = baseline_group if (baseline_group is not None and i == 0) else group
        dw = ordered_day_windows(sess_sub, day_col, window_col)
        xs = [x0 + j for j in range(len(dw))]
        blocks.append((sess, sess_group, dw, xs))
        x0 = xs[-1] + 1 + gap if xs else x0

    for sess, sess_group, dw, xs in blocks:
        order = [g for g in GROUP_ORDERS[sess_group] if g in sub_sex[sess_group].unique()]
        palette = GROUP_PALETTES[sess_group](sex)
        for grp in order:
            color = palette[grp]
            if sess_group == 'condition':
                _, treat = grp.split('_', 1)
                style = TREATMENT_STYLE.get(treat, dict(marker='o', ls='-', alpha=0.9))
            else:
                style = dict(marker='o', ls='-', alpha=0.9)
            label = f'{grp} ({sex_suffix})' if sex_suffix and _is_sex_variant_group(grp) else grp
            d = sub_sex[(sub_sex[sess_group] == grp) & (sub_sex[time_point_col] == sess)]
            g_stats = d.groupby([day_col, window_col])[col].agg(['mean', 'sem']).reindex(dw)
            ax.errorbar(xs, g_stats['mean'], yerr=g_stats['sem'], color=color, capsize=3, lw=2,
                        markersize=5, label=label, **style)

    all_xs, all_labels = [], []
    for sess, sess_group, dw, xs in blocks:
        all_xs += xs
        all_labels += [w for _, w in dw]
        days_seen = sorted({d for d, _ in dw})
        block_start = {d: next(i for i, (dd, _) in enumerate(dw) if dd == d) for d in days_seen}
        boundaries = [block_start[d] for d in days_seen] + [len(dw)]
        for i, d in enumerate(days_seen):
            lo, hi = xs[boundaries[i]] - 0.5, xs[boundaries[i + 1] - 1] + 0.5
            if i % 2 == 1:
                ax.axvspan(lo, hi, color=COLOR_MUTED, alpha=0.06, zorder=0)
            if i > 0:
                ax.axvline(lo, color=COLOR_MUTED, lw=0.5, alpha=0.3, zorder=0)
            if len(days_seen) > 1:
                ax.text((lo + hi) / 2, 0.97, f'D{d}', transform=ax.get_xaxis_transform(),
                        ha='center', va='top', fontsize=6, color=COLOR_MUTED)

    if len(blocks) == 2:
        break_x = (blocks[0][3][-1] + blocks[1][3][0]) / 2
        ax.axvline(break_x, color=COLOR_INK, lw=1.3, alpha=0.5, zorder=2)
        ax.text(break_x, 0.5, ' treatment ', transform=ax.get_xaxis_transform(), rotation=90,
                ha='center', va='center', fontsize=6.5, color=COLOR_INK, alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75))

    if label_style:
        all_labels = [format_window_label(w, style=label_style,
                                           active_start_hour=active_start_hour)
                       for w in all_labels]
        axis_name = ('time since active phase start' if label_style == 'relative'
                      else 'time_window (ZT, h)')
    else:
        axis_name = 'time_window (ZT, h)'
    multi_day_axis = any(len({d for d, _ in dw}) > 1 for _, _, dw, _ in blocks)
    tick_fontsize = 6
    ax.set_xticks(all_xs)
    ax.set_xticklabels(all_labels, fontsize=tick_fontsize, rotation=90)

    # The tick labels are rotated 90 degrees, so they eat a vertical strip below the axis whose
    # height depends on the LONGEST label ('90-120min', '12.0-12.5', ...). The block captions have
    # to clear that strip. Everything here is in POINTS, deliberately: a grid calls tight_layout()
    # after these panels are built, which rescales the axes and so invalidates any offset
    # expressed as an axes fraction - but font sizes, and therefore this strip, don't rescale.
    try:
        renderer = ax.figure.canvas.get_renderer()
        strip_px = max(t.get_window_extent(renderer).height for t in ax.get_xticklabels())
        tick_strip_pts = strip_px * 72.0 / ax.figure.dpi
    except Exception:
        tick_strip_pts = max((len(l) for l in all_labels), default=0) * tick_fontsize * 0.62

    # labelpad is padding ON TOP OF matplotlib's automatic clearance of the tick labels, so it
    # must NOT include tick_strip_pts again - it only has to clear the caption row (8pt text hung
    # 6pt below the strip), otherwise the axis label drifts far below the plot.
    ax.set_xlabel(axis_name + (', by day' if multi_day_axis else ''), fontsize=7.5,
                  labelpad=17 if caption_loc == 'below' else 4)

    if caption_loc == 'below':
        block_captions = {**SESSION_BLOCK_LABELS, **(session_labels or {})}
        for sess, _, _, xs in blocks:
            if not xs:
                continue
            ax.annotate(block_captions.get(sess, sess),
                        xy=((xs[0] + xs[-1]) / 2, 0), xycoords=('data', 'axes fraction'),
                        xytext=(0, -(tick_strip_pts + 6)), textcoords='offset points',
                        ha='center', va='top', fontsize=8, color=COLOR_INK, fontweight='bold')
    ax.axhline(0, color=COLOR_MUTED, lw=0.5, ls=':', zorder=0)
    ax.set_title(_domain_label(feature), fontsize=10.5)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            seen.setdefault(l, h)
        ax.legend(seen.values(), seen.keys(), fontsize=6.5, loc='best')
    sns.despine(ax=ax)
    return ax


def plot_session_trajectory_grid(data, features, sexes, group='condition', baseline_group=None,
                                  session_order=('baseline', 'MDMA'), pair_sexes=False,
                                  max_cols=None, legend_loc='top', value_suffix='',
                                  label_style=None, active_start_hour=12, ylabel=None,
                                  session_labels=None, caption_loc='below', title=None,
                                  save_path=None, show=True, dpi=150):
    """Grid of plot_session_trajectory() panels - the full baseline -> post-injection
    trajectory for each feature/domain in `features` - meant for CORE_DOMAIN_FEATURES-sized
    lists (12 domains) where one row would be unreadably wide.

    `sexes` - a single sex string (one row of panels, the original single-sex behavior) or
    a tuple of sexes. Domain scores are z-scored per sex (and the ELA color itself is
    sex-specific - see SEX_ELA_COLORS), so comparing sexes only makes sense on this
    z-scored axis, which pair_sexes below takes advantage of.

    baseline_group: passed straight through to plot_session_trajectory() - set to 'background'
        to collapse the baseline block of every panel to the 2-way CTRL/ELA split instead of
        following `group` (typically 'condition', 4-way) there too. See plot_session_trajectory's
        docstring for why that split makes more sense at baseline. None (default) keeps `group`
        uniform across the whole trajectory.

    Two layouts when `sexes` has more than one entry:
    - pair_sexes=False (default): one row per sex, columns wrap into extra row-groups
      (`max_cols` features per group, or an automatic sqrt(n)-based choice) when
      `features` is long.
    - pair_sexes=True: requires exactly two sexes. Each feature gets its male/female
      panels side by side with a shared y-axis scale, y tick labels suppressed on the
      second panel. Feature-pairs wrap into a grid the same way when there are many
      features.

    A single figure-level legend (deduped across panels) replaces the old per-panel corner
    legend - legend_loc='top'/'bottom' places it above/below the grid, or None to omit it.

    Input: `data` - output of load_temporal_data(..., time_points=session_order, keep_day=True).

    Returns: (fig, axes)
    """
    # caption_loc='title' folds the session names into each panel title
    # ('Exploration (baseline | after treatment)') instead of writing them under the blocks -
    # more compact, and keeps the x-axis label tight against the ticks.
    _caps = {**SESSION_BLOCK_LABELS, **(session_labels or {})}
    title_suffix = (' (' + ' | '.join(_caps.get(s_, s_) for s_ in session_order) + ')'
                     if caption_loc == 'title' else '')

    sexes = [sexes] if isinstance(sexes, str) else list(sexes)
    n_feat = len(features)
    panel_w = 4.4

    if pair_sexes:
        if len(sexes) != 2:
            raise ValueError('pair_sexes=True requires exactly two sexes')
        sex_left, sex_right = sexes
        ncols_blocks = min(n_feat, max_cols or _auto_max_cols(n_feat, small_threshold=3, min_cols=2))
        nrows_blocks = int(np.ceil(n_feat / ncols_blocks))

        fig, axes = plt.subplots(nrows_blocks, ncols_blocks * 2,
                                  figsize=(panel_w * ncols_blocks * 2, 3.6 * nrows_blocks),
                                  squeeze=False)

        for i, feat in enumerate(features):
            row, col_block = divmod(i, ncols_blocks)
            ax_left, ax_right = axes[row, col_block * 2], axes[row, col_block * 2 + 1]
            traj_kw = dict(group=group, baseline_group=baseline_group,
                            session_order=session_order, show_legend=False,
                            value_suffix=value_suffix, label_style=label_style,
                            active_start_hour=active_start_hour, session_labels=session_labels,
                            caption_loc=caption_loc)
            plot_session_trajectory(data, feat, sex_left, ax_left, sex_suffix=sex_left, **traj_kw)
            plot_session_trajectory(data, feat, sex_right, ax_right, sex_suffix=sex_right, **traj_kw)

            ylo = min(ax_left.get_ylim()[0], ax_right.get_ylim()[0])
            yhi = max(ax_left.get_ylim()[1], ax_right.get_ylim()[1])
            ax_left.set_ylim(ylo, yhi)
            ax_right.set_ylim(ylo, yhi)
            ax_right.tick_params(labelleft=False)

            domain_label = _domain_label(feat)
            ax_left.set_title(f'{domain_label} ({sex_left}){title_suffix}', fontsize=8.5)
            ax_right.set_title(f'{domain_label} ({sex_right}){title_suffix}', fontsize=8.5)

        for i in range(n_feat, nrows_blocks * ncols_blocks):
            row, col_block = divmod(i, ncols_blocks)
            axes[row, col_block * 2].axis('off')
            axes[row, col_block * 2 + 1].axis('off')

    else:
        ncols = min(n_feat, max_cols or _auto_max_cols(n_feat, small_threshold=6, min_cols=3))
        n_groups = int(np.ceil(n_feat / ncols))
        nrows = n_groups * len(sexes)

        fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, 3.6 * nrows), squeeze=False)

        for grp_i in range(n_groups):
            feats_in_group = features[grp_i * ncols:(grp_i + 1) * ncols]
            for row_offset, sex in enumerate(sexes):
                r = grp_i * len(sexes) + row_offset
                for col, feat in enumerate(feats_in_group):
                    ax = axes[r, col]
                    plot_session_trajectory(data, feat, sex, ax, group=group,
                                             baseline_group=baseline_group,
                                             session_order=session_order, show_legend=False,
                                             sex_suffix=sex if len(sexes) > 1 else None,
                                             value_suffix=value_suffix, label_style=label_style,
                                             active_start_hour=active_start_hour,
                                             session_labels=session_labels,
                                             caption_loc=caption_loc)
                    ax.set_title(f'{_domain_label(feat)}{title_suffix}' if row_offset == 0
                                  else '', fontsize=9)
                    if col == 0:
                        ax.set_ylabel(f'{sex}\n{ylabel}' if ylabel else
                                       (sex if len(sexes) > 1 else ''), fontsize=9)
                for col in range(len(feats_in_group), ncols):
                    axes[r, col].axis('off')

    top_pad, bottom_pad, title_y, legend_y = _legend_layout(legend_loc, title)
    plt.tight_layout(rect=(0, bottom_pad, 1, 1 - top_pad))
    default_title = (f'{sexes[0]}: baseline -> post-injection domain-score trajectory'
                      if len(sexes) == 1 else
                      'Baseline -> post-injection domain-score trajectory, ' + ' vs '.join(sexes))
    fig.suptitle(title or default_title, color=COLOR_INK, fontsize=13, fontweight='bold', y=title_y)
    _draw_shared_legend(fig, axes.flatten(), legend_loc, legend_y)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes


#%%
# ============================================================================
# STANDALONE SMOKE TEST - exercises both call sites (step 6 baseline/background, step 7
# post-injection/condition) at a couple of resolutions, without saving. Only runs when this file
# is executed on its own (`python temporal_dynamics.py`), not when imported.
# ============================================================================
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from domain_scores import CORE_DOMAIN_FEATURES

    DATA_PATH = '../behavior_dataset/final/QC_table'

    # Step 6 style: baseline only, split by background, 3h resolution. All 12 core domains -
    # exercises the auto-wrapping layout (row-per-sex, wrapped into row-groups) and the shared
    # top legend.
    baseline_3h = load_temporal_data('3h', time_points='baseline', data_path=DATA_PATH)
    plot_temporal_grid(baseline_3h, CORE_DOMAIN_FEATURES, group='background',
                        ylabel='z-scored domain composite', legend_loc='top',
                        title='BASELINE: ELA vs CTRL, domain scores across the active phase '
                              '(3h bins, smoke test)')

    # Same data, pair_sexes=True: each domain's male/female panels side by side on a shared
    # y-axis scale, legend at the bottom instead of the top.
    plot_temporal_grid(baseline_3h, CORE_DOMAIN_FEATURES[:6], group='background',
                        ylabel='z-scored domain composite', pair_sexes=True, legend_loc='bottom',
                        title='BASELINE: ELA vs CTRL, male vs female side by side '
                              '(3h bins, smoke test)')

    # Step 7 style: post-injection only, split by condition (4-way), 6h resolution.
    mdma_6h = load_temporal_data('6h', time_points='MDMA', data_path=DATA_PATH)
    plot_temporal_grid(mdma_6h, CORE_DOMAIN_FEATURES[:4], group='condition',
                        ylabel='z-scored domain composite',
                        title='POST-INJECTION: absolute domain scores by condition '
                              '(6h bins, smoke test)')

    # Baseline + MDMA overlaid in one panel, split by background, 1h resolution, one raw feature.
    both_1h = load_temporal_data('1h', extra_features=['speeding_duration_fraction'],
                                  time_points=['baseline', 'MDMA'], data_path=DATA_PATH)
    fig, ax = plt.subplots(figsize=(5, 4.3))
    plot_temporal(both_1h, 'speeding_duration_fraction', sex='male', ax=ax, group='background')
    ax.set_title('Speeding, duration fraction (male, 1h bins, smoke test)')
    plt.show()

    # Day selection: day 1 only vs. day2+3 average (mirrors run_analysis.py's
    # baseline_active_day1 / baseline_active_day23), both still collapsed to one within-day shape.
    baseline_day1 = load_temporal_data('3h', time_points='baseline', days=1, data_path=DATA_PATH)
    baseline_day23 = load_temporal_data('3h', time_points='baseline', days=[2, 3], data_path=DATA_PATH)
    plot_temporal_grid(baseline_day1, CORE_DOMAIN_FEATURES[:3], group='background',
                        title='BASELINE day 1 only (3h bins, smoke test)')
    plot_temporal_grid(baseline_day23, CORE_DOMAIN_FEATURES[:3], group='background',
                        title='BASELINE day 2+3 average (3h bins, smoke test)')

    # keep_day=True: actual continuous 3-day trajectory (day left as its own axis, not averaged).
    baseline_all_days = load_temporal_data('3h', time_points='baseline', keep_day=True,
                                            data_path=DATA_PATH)
    plot_temporal_grid(baseline_all_days, CORE_DOMAIN_FEATURES[:3], group='background',
                        title='BASELINE, continuous days 1-3 trajectory (3h bins, smoke test)')

    # Full baseline -> post-injection trajectory, split by condition, 6h resolution, all core
    # domains wrapped into a grid, one row per sex.
    full_traj_6h = load_temporal_data('6h', time_points=['baseline', 'MDMA'], keep_day=True,
                                       data_path=DATA_PATH)
    plot_session_trajectory_grid(
        full_traj_6h, CORE_DOMAIN_FEATURES, sexes=('female', 'male'), group='condition',
        title='BASELINE -> POST-INJECTION domain-score trajectory by condition '
              '(6h bins, smoke test)')

    # Same data, pair_sexes=True: each domain's male/female panels side by side on a shared
    # y-axis scale, legend at the bottom.
    plot_session_trajectory_grid(
        full_traj_6h, CORE_DOMAIN_FEATURES[:6], sexes=('female', 'male'), group='condition',
        pair_sexes=True, legend_loc='bottom',
        title='BASELINE -> POST-INJECTION trajectory, male vs female side by side '
              '(6h bins, smoke test)')

    # baseline_group='background': baseline block collapses to CTRL vs ELA (2 lines) while the
    # post-injection block keeps the full 4-way condition split.
    plot_session_trajectory_grid(
        full_traj_6h, CORE_DOMAIN_FEATURES[:4], sexes=('female', 'male'), group='condition',
        baseline_group='background',
        title='BASELINE (CTRL/ELA only) -> POST-INJECTION (by condition) trajectory '
              '(6h bins, smoke test)')
