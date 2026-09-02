"""
aggregate.py - flexible aggregation & comparison-table builders.

These replace the earlier hardcoded "mouse_level_session_means / baseline_box_paired /
treatment_deltas" functions with generic versions parameterized by whatever grouping you want
(per-mouse-per-session, per-mouse-per-day, per-mouse-per-time-window, etc.) - same functions
work whether you're collapsing days 1-3 into one baseline number, or keeping each day separate,
or working with 1h/3h-resolution data.
"""
import numpy as np
import pandas as pd


def aggregate_to_mouse(df, group_cols, features, agg='mean'):
    """Collapse repeated rows (e.g. multiple days, or multiple time-bins) into one row per
    group_cols combination. This is the single generic building block for "session means",
    "day-level values", "per-time-window values", etc. - just change group_cols.

    Examples:
        # per mouse, per session (collapses days 1-3 together) - what earlier code called
        # "mouse_level_session_means":
        aggregate_to_mouse(df, ['mouse_ID','sex','age','background','treatment','box_ID',
                                 'time_point'], features)

        # per mouse, per day (keeps days separate) - for "explore by day":
        aggregate_to_mouse(df, ['mouse_ID','sex','age','background','treatment','box_ID',
                                 'day','time_point'], features)

        # per mouse, per fine time-window (for "explore by time window"):
        aggregate_to_mouse(df, ['mouse_ID','sex','age','background','treatment','box_ID',
                                 'time_window'], features)
    """
    return df.groupby(group_cols, as_index=False)[features].agg(agg)


def box_paired_wide(df, features, pair_col='background', pair_values=('ELA', 'CTRL'),
                     extra_group_cols=('sex',)):
    """Average the two pair_values (e.g. 2 ELA + 2 CTRL mice) within each box separately,
    producing a wide table: one row per box (x any extra_group_cols, e.g. per box per day),
    columns <feature>_<pair_value> for each value in pair_values.

    This generalizes the earlier "baseline_box_paired" (which only worked for background at
    the collapsed-baseline level) to work for any grouping, e.g. pass extra_group_cols=
    ('sex','day') to get day-resolved box-pairs, or pair_col='treatment', pair_values=
    ('MDMA','saline') for a box-level treatment comparison (note: treatment isn't 2-vs-2
    within a box the way background is, so that use case is usually better served by
    box_level_means below - included here for completeness).
    """
    group_cols = list(extra_group_cols) + ['box_ID', pair_col]
    box_avg = df.groupby(group_cols, as_index=False)[features].mean()
    idx_cols = list(extra_group_cols) + ['box_ID']
    wide = box_avg.pivot(index=idx_cols, columns=pair_col, values=features)
    wide.columns = [f'{feat}_{val}' for feat, val in wide.columns]
    return wide.reset_index()


def box_level_means(df, features, group_cols=('sex', 'box_ID')):
    """Simple box-level average (not paired) - one row per box (x extra grouping), used e.g.
    for treatment (box-level factor, not 2-vs-2 within box like background)."""
    return df.groupby(list(group_cols), as_index=False)[features].mean()


def mouse_level_deltas(df, features, id_cols, session_col='time_point',
                        session_values=('baseline', 'MDMA'), suffix='delta'):
    """Per mouse (or per mouse-per-day, etc, depending on id_cols): delta = session_values[1] -
    session_values[0]. Drops rows missing either session. id_cols should uniquely identify the
    unit you want one delta per (e.g. ['mouse_ID','sex','age','background','treatment','box_ID']
    for one delta per mouse collapsed across days, or add 'day' to get per-day deltas).

    Generalizes the earlier hardcoded "treatment_deltas" (which only did baseline->MDMA
    collapsed across days) to work for any pair of session_col values and any id_cols grouping.
    """
    piv = df.pivot_table(index=id_cols, columns=session_col, values=features)
    piv.columns = [f'{feat}__{sess}' for feat, sess in piv.columns]
    a, b = session_values
    need_cols = [f'{feat}__{a}' for feat in features] + [f'{feat}__{b}' for feat in features]
    piv = piv.dropna(subset=need_cols, how='any')
    for feat in features:
        piv[f'{feat}__{suffix}'] = piv[f'{feat}__{b}'] - piv[f'{feat}__{a}']
    return piv.reset_index()
