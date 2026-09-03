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
'MDMA') and keep_day=True for this.

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


def load_temporal_data(resolution, domain_defs=None, extra_features=None, id_cols=None,
                        phase='active', time_points=None, days=None, keep_day=False,
                        reference_time_point='baseline', data_path=None, female_path=None,
                        male_path=None, verbose=True):
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

    df = load_data(female_path=female_path, male_path=male_path, verbose=verbose)
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
    """Sort time_window labels ('12-18','18-24',... or '0-3','3-6',...) by their numeric start
    hour. Valid within a single phase (windows returned by load_temporal_data don't wrap
    midnight within a phase - see prep.py)."""
    return sorted(data[window_col].unique().tolist(), key=lambda w: int(w.split('-')[0]))


def ordered_day_windows(data, day_col='day', window_col='time_window'):
    """Sort (day, time_window) combinations for a continuous multi-day x-axis: primarily by day,
    then by each window's numeric start hour within that day. Input: `data` with load_temporal_
    data(..., keep_day=True)'s 'day' column present."""
    combos = data[[day_col, window_col]].drop_duplicates().itertuples(index=False, name=None)
    return sorted(combos, key=lambda dw: (dw[0], int(dw[1].split('-')[0])))


def plot_temporal(data, feature, sex, ax, group='background', window_col='time_window',
                   day_col='day', show_legend=True):
    """One feature's (or domain score's) time-course across time_window, for one sex, split by
    `group` - 'background' (CTRL vs ELA, 2 lines) or 'condition' (the 4 background x treatment
    groups, 4 lines).

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
    sub_sex = data[data.sex == sex]
    order = [g for g in GROUP_ORDERS[group] if g in sub_sex[group].unique()]
    palette = GROUP_PALETTES[group](sex)
    time_points = [t for t in TIME_POINT_STYLE if t in sub_sex['time_point'].unique()] + \
        [t for t in sub_sex['time_point'].unique() if t not in TIME_POINT_STYLE]

    multi_day = day_col in sub_sex.columns and sub_sex[day_col].nunique() > 1
    if multi_day:
        day_windows = ordered_day_windows(sub_sex, day_col, window_col)
        x = np.arange(len(day_windows))
        xticklabels = [w.split('-')[1] for _, w in day_windows]
    else:
        windows = ordered_windows(sub_sex, window_col)
        x = np.arange(len(windows))
        xticklabels = [w.split('-')[1] for w in windows]

    for grp in order:
        color = palette[grp]
        if group == 'condition':
            _, treat = grp.split('_', 1)
            cond_style = TREATMENT_STYLE.get(treat, dict(marker='o', ls='-', alpha=0.9))
        else:
            cond_style = None
        for tp in time_points:
            d = sub_sex[(sub_sex[group] == grp) & (sub_sex['time_point'] == tp)]
            if multi_day:
                g = d.groupby([day_col, window_col])[feature].agg(['mean', 'sem']).reindex(day_windows)
            else:
                g = d.groupby(window_col)[feature].agg(['mean', 'sem']).reindex(windows)
            style = cond_style or TIME_POINT_STYLE.get(tp, dict(marker='o', ls='-', alpha=0.9))
            label = grp if len(time_points) == 1 else f'{grp} ({tp})'
            ax.errorbar(x, g['mean'], yerr=g['sem'], color=color, capsize=3, lw=2, markersize=5,
                        label=label, **style)

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

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=6.5 if multi_day else 8,
                        rotation=90 if multi_day else 0)
    ax.set_xlabel('Hour (end of each time bin)' + (', by day' if multi_day else ''), fontsize=8.5)
    ax.axhline(0, color=COLOR_MUTED, lw=0.5, ls=':', zorder=0)
    ax.set_title(_domain_label(feature), fontsize=10.5)
    if show_legend:
        ax.legend(fontsize=7, loc='best')
    sns.despine(ax=ax)
    return ax


def plot_temporal_grid(data, features, sexes=('female', 'male'), group='background',
                        window_col='time_window', day_col='day', ylabel=None, title=None,
                        save_path=None, show=True, dpi=150):
    """Reference grid: one row per sex, one column per feature/domain, each panel via
    plot_temporal(). Input: `data` - output of load_temporal_data(). Panels widen automatically
    when `data` is multi-day (load_temporal_data(..., keep_day=True)) to fit the longer x-axis.

    Returns: (fig, axes)
    """
    multi_day = day_col in data.columns and data[day_col].nunique() > 1
    panel_w = 4.2 if multi_day else 2.8
    fig, axes = plt.subplots(len(sexes), len(features),
                              figsize=(panel_w * len(features), 3.9 * len(sexes)), squeeze=False)

    for row, sex in enumerate(sexes):
        for col, feat in enumerate(features):
            ax = axes[row, col]
            plot_temporal(data, feat, sex, ax, group=group, window_col=window_col,
                           day_col=day_col, show_legend=(row == 0 and col == len(features) - 1))
            ax.set_title(_domain_label(feat) if row == 0 else '', fontsize=9)
            ax.set_ylabel(f'{sex}\n{ylabel}' if (col == 0 and ylabel) else
                           (sex if col == 0 else ''), fontsize=9)

    if title:
        fig.suptitle(title, color=COLOR_INK, fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes


def plot_session_trajectory(data, feature, sex, ax, group='condition',
                             session_order=('baseline', 'MDMA'), time_point_col='time_point',
                             day_col='day', window_col='time_window', gap=1.2, show_legend=True):
    """One feature's (or domain score's) FULL trajectory across BOTH sessions in a single panel:
    baseline days 1-3 on the left, a visual break at the treatment boundary, then post-injection
    (MDMA) days 1-3 on the right - same within-day (day, time_window) x-axis logic as
    plot_temporal()'s multi-day mode, just placed in two side-by-side blocks (one per session)
    instead of overlaid on the same x-axis, since the two sessions are different points in time,
    not two readings of the same window.

    Input: `data` - output of load_temporal_data(..., time_points=session_order, keep_day=True).
    Z-scoring should be baseline-referenced (load_temporal_data's default) so 0 means "typical
    baseline" on BOTH sides of the break, making the jump at the treatment boundary meaningful.

    group='condition' (default) is the natural choice here since both treatment arms are in play
    across the break - styled via TREATMENT_STYLE (saline fainter/dashed, MDMA solid), color by
    background, CONSTANT across the break so e.g. the ELA_MDMA line reads as the same group on
    both sides, just discontinuous. group='background' also works (2 lines, plain solid) if you
    don't need the treatment split.

    Returns: ax
    """
    if group not in GROUP_ORDERS:
        raise ValueError(f"group must be one of {list(GROUP_ORDERS)}, got {group!r}")
    sub_sex = data[data.sex == sex]
    order = [g for g in GROUP_ORDERS[group] if g in sub_sex[group].unique()]
    palette = GROUP_PALETTES[group](sex)

    blocks = []  # (session, day_windows, xs)
    x0 = 0
    for sess in session_order:
        sess_sub = sub_sex[sub_sex[time_point_col] == sess]
        if not len(sess_sub):
            continue
        dw = ordered_day_windows(sess_sub, day_col, window_col)
        xs = [x0 + i for i in range(len(dw))]
        blocks.append((sess, dw, xs))
        x0 = xs[-1] + 1 + gap if xs else x0

    for grp in order:
        color = palette[grp]
        if group == 'condition':
            _, treat = grp.split('_', 1)
            style = TREATMENT_STYLE.get(treat, dict(marker='o', ls='-', alpha=0.9))
        else:
            style = dict(marker='o', ls='-', alpha=0.9)
        for sess, dw, xs in blocks:
            d = sub_sex[(sub_sex[group] == grp) & (sub_sex[time_point_col] == sess)]
            g = d.groupby([day_col, window_col])[feature].agg(['mean', 'sem']).reindex(dw)
            ax.errorbar(xs, g['mean'], yerr=g['sem'], color=color, capsize=3, lw=2, markersize=5,
                        label=grp, **style)

    all_xs, all_labels = [], []
    for sess, dw, xs in blocks:
        all_xs += xs
        all_labels += [w.split('-')[1] for _, w in dw]
        days_seen = sorted({d for d, _ in dw})
        block_start = {d: next(i for i, (dd, _) in enumerate(dw) if dd == d) for d in days_seen}
        boundaries = [block_start[d] for d in days_seen] + [len(dw)]
        for i, d in enumerate(days_seen):
            lo, hi = xs[boundaries[i]] - 0.5, xs[boundaries[i + 1] - 1] + 0.5
            if i % 2 == 1:
                ax.axvspan(lo, hi, color=COLOR_MUTED, alpha=0.06, zorder=0)
            if i > 0:
                ax.axvline(lo, color=COLOR_MUTED, lw=0.5, alpha=0.3, zorder=0)
            ax.text((lo + hi) / 2, 0.97, f'D{d}', transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=6, color=COLOR_MUTED)

    if len(blocks) == 2:
        break_x = (blocks[0][2][-1] + blocks[1][2][0]) / 2
        ax.axvline(break_x, color=COLOR_INK, lw=1.3, alpha=0.5, zorder=2)
        ax.text(break_x, 0.5, ' treatment ', transform=ax.get_xaxis_transform(), rotation=90,
                ha='center', va='center', fontsize=6.5, color=COLOR_INK, alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75))

    ax.set_xticks(all_xs)
    ax.set_xticklabels(all_labels, fontsize=6, rotation=90)
    ax.set_xlabel(f'Hour, by day ({" | ".join(session_order)})', fontsize=7.5)
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


def plot_session_trajectory_grid(data, features, sex, group='condition', ncols=3,
                                  session_order=('baseline', 'MDMA'), title=None, save_path=None,
                                  show=True, dpi=150):
    """One figure for ONE sex: the full baseline -> post-injection trajectory
    (plot_session_trajectory()) for each feature/domain in `features`, wrapped into a grid
    (`ncols` columns) rather than a single wide row - meant for CORE_DOMAIN_FEATURES-sized lists
    (12 domains) where one row per sex would be unreadably wide. Call once per sex (domain scores
    are z-scored per sex, and the ELA color itself is sex-specific - see SEX_ELA_COLORS) for two
    separate figures.

    Input: `data` - output of load_temporal_data(..., time_points=session_order, keep_day=True).

    Returns: (fig, axes)
    """
    n = len(features)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, feat in enumerate(features):
        plot_session_trajectory(data, feat, sex, axes_flat[i], group=group,
                                 session_order=session_order, show_legend=(i == 0))
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis('off')

    fig.suptitle(title or f'{sex}: baseline -> post-injection domain-score trajectory',
                 color=COLOR_INK, fontsize=13, fontweight='bold', y=1.0)
    plt.tight_layout()

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

    # Step 6 style: baseline only, split by background, 3h resolution.
    baseline_3h = load_temporal_data('3h', time_points='baseline', data_path=DATA_PATH)
    plot_temporal_grid(baseline_3h, CORE_DOMAIN_FEATURES[:4], group='background',
                        ylabel='z-scored domain composite',
                        title='BASELINE: ELA vs CTRL, domain scores across the active phase '
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
    # domains wrapped into a grid, one figure per sex.
    full_traj_6h = load_temporal_data('6h', time_points=['baseline', 'MDMA'], keep_day=True,
                                       data_path=DATA_PATH)
    for sex in ['female', 'male']:
        plot_session_trajectory_grid(
            full_traj_6h, CORE_DOMAIN_FEATURES, sex=sex, group='condition', ncols=3,
            title=f'{sex}: BASELINE -> POST-INJECTION domain-score trajectory by condition '
                  '(6h bins, smoke test)')
