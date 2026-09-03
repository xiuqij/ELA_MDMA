"""
plot_domain_score_by_day.py - day-resolved ELA vs CTRL trajectory plot for domain scores,
with per-day box-paired effect-size annotations.

Reusable plotting function meant to be imported wherever you already have a day-resolved
box-paired wide table (aggregate.box_paired_wide(..., extra_group_cols=('sex','day'))) and
want to visualize how a domain score's ELA-vs-CTRL gap evolves across baseline days 1-3 (or
any other 'day'-like column) - one panel per feature/sex, thin gray lines for individual
box-pairs, bold mean+/-SEM trajectories, and a per-day Cohen's dz (+ significance stars)
annotation computed via stats_utils.paired_effect_table.

Usage - import into run_analysis.py (or any other script) right after building `wide_by_day`
in step 5 ("EXPLORE BY DAY"):

    from plot_domain_score_by_day import plot_domain_score_by_day, plot_domain_scores_by_day_grid

    fig, ax = plt.subplots(figsize=(3.2, 4.3))
    plot_domain_score_by_day(wide_by_day, 'nest_occupancy_score', 'male', ax=ax)

    # or, several domains x both sexes in one reference grid:
    plot_domain_scores_by_day_grid(
        wide_by_day, CORE_DOMAIN_FEATURES,
        title='BASELINE: ELA vs CTRL, domain scores by day (box-paired)',
        save_path='domain_scores_by_day.png',
    )

Standalone: `python plot_domain_score_by_day.py` reproduces run_analysis.py steps 1-5 (day-level)
on its own and shows an example grid, no save - smoke test.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stats_utils import paired_effect_table

# Palette consistent with plot_effect_table.py (dataviz skill diverging pair).
COLOR_CTRL = '#898781'
COLOR_MUTED = '#898781'
COLOR_INK = '#0b0b0b'
# ELA line color per sex - matches the categorical convention used elsewhere for sex panels.
SEX_COLORS = {'female': '#e34948', 'male': '#2a78d6'}


def _domain_label(feature):
    return feature.replace('_score', '').replace('_', ' ').capitalize()


def plot_domain_score_by_day(wide_by_day, feature, sex, ax=None, day_col='day',
                              show_individual_boxes=True, annotate_stats=True):
    """Plot one domain feature's ELA-vs-CTRL trajectory across days for one sex.

    Args:
        wide_by_day: output of aggregate.box_paired_wide(mouse_by_day, features,
            extra_group_cols=('sex', day_col)) - one row per box per day, with columns
            '<feature>_ELA' / '<feature>_CTRL'.
        feature: domain-score column name to plot (e.g. 'nest_occupancy_score').
        sex: 'female' or 'male' - filters wide_by_day.
        ax: matplotlib axes to draw into (created if None).
        day_col: name of the day-like grouping column (default 'day').
        show_individual_boxes: draw thin per-box-pair lines behind the mean+/-SEM trajectory.
        annotate_stats: draw a per-day Cohen's dz (+ significance stars, paired t-test) line
            below the axes, computed via stats_utils.paired_effect_table (one call per day -
            stars are raw per-day p-values, not corrected across days).

    Returns: ax
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 4.3))

    col_ela, col_ctrl = f'{feature}_ELA', f'{feature}_CTRL'
    sub = wide_by_day[wide_by_day['sex'] == sex]
    days = sorted(sub[day_col].unique())

    if show_individual_boxes:
        for _, row in sub.iterrows():
            if pd.isna(row[col_ela]) or pd.isna(row[col_ctrl]):
                continue
            ax.plot(row[day_col] + np.array([-0.08, 0.08]), [row[col_ela], row[col_ctrl]],
                    color=COLOR_MUTED, alpha=0.25, lw=1, zorder=1)

    for label, col, style in [('CTRL', col_ctrl, dict(color=COLOR_CTRL, marker='o')),
                               ('ELA', col_ela, dict(color=SEX_COLORS.get(sex, '#e34948'), marker='s'))]:
        means = sub.groupby(day_col)[col].mean()
        sems = sub.groupby(day_col)[col].sem()
        ax.errorbar(days, means.loc[days], yerr=sems.loc[days], label=label, capsize=4, lw=2.2,
                    markersize=7, **style, zorder=3)

    ax.set_xticks(days)
    ax.set_xlabel(day_col.capitalize())
    ax.set_title(_domain_label(feature), fontsize=10.5)

    if annotate_stats:
        stat_txt = []
        for day in days:
            day_sub = sub[sub[day_col] == day]
            res = paired_effect_table(day_sub, [feature])
            if len(res):
                dz, p = res.loc[0, 'cohen_dz'], res.loc[0, 'p_ttest']
                star = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
                stat_txt.append(f'{day}: {dz:+.1f}{star}')
        if stat_txt:
            ax.text(0.5, -0.28, '  '.join(stat_txt), transform=ax.transAxes, ha='center',
                    fontsize=7.5, style='italic', color=COLOR_INK)

    return ax


def plot_domain_scores_by_day_grid(wide_by_day, features, sexes=('female', 'male'), day_col='day',
                                    show_individual_boxes=False, annotate_stats=True,
                                    title=None, save_path=None, show=True, dpi=150):
    """Reference grid: one row per sex, one column per feature, each panel via
    plot_domain_score_by_day(). Shared legend, shared y-label per row.

    Args:
        wide_by_day: see plot_domain_score_by_day().
        features: list of domain-score column names, left-to-right column order.
        sexes: top-to-bottom row order (default female, male).
        save_path: if given, saves the figure there (parent dir created if needed).
        show: whether to call plt.show() (default True; set False for non-interactive use).

    Returns: (fig, axes)
    """
    fig, axes = plt.subplots(len(sexes), len(features),
                              figsize=(2.6 * len(features), 3.9 * len(sexes)), squeeze=False)

    for row, sex in enumerate(sexes):
        for col, feat in enumerate(features):
            ax = axes[row, col]
            plot_domain_score_by_day(wide_by_day, feat, sex, ax=ax, day_col=day_col,
                                      show_individual_boxes=show_individual_boxes,
                                      annotate_stats=annotate_stats)
            ax.set_title(_domain_label(feat), fontsize=9) if row == 0 else ax.set_title('')
            if col == 0:
                ax.set_ylabel(f'{sex}\nz-score', fontsize=9)

    axes[0, -1].legend(fontsize=7, loc='upper right')
    if title:
        fig.suptitle(title, color=COLOR_INK, fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()

    return fig, axes


#%%
# ============================================================================
# STANDALONE SMOKE TEST - reproduces run_analysis.py steps 1-5 (day-level) and calls
# plot_domain_scores_by_day_grid() directly, without saving. Only runs when this file is
# executed on its own (`python plot_domain_score_by_day.py`), not when imported.
# ============================================================================
if __name__ == '__main__':
    import sys
    sys.path.insert(0, 'lib')

    from prep import load_data, filter_data
    from aggregate import aggregate_to_mouse, box_paired_wide
    from domain_scores import compute_domain_scores, CORE_DOMAINS, CORE_DOMAIN_FEATURES, ALL_DOMAINS

    MOUSE_ID_COLS = ['mouse_ID', 'sex', 'age', 'background', 'treatment', 'box_ID']
    DATA_PATH = '/Users/xiuqi.ji/Library/CloudStorage/OneDrive-KarolinskaInstitutet/MDMA_ELA/social_box_2026/github_repo/behavior_dataset/final/QC_table'

    df = load_data(female_path=f'{DATA_PATH}/female_12h_filtered.csv',
                    male_path=f'{DATA_PATH}/male_12h_filtered.csv')
    baseline_active = filter_data(df, phase='active', time_point='baseline')

    raw_features = sorted({f for spec in ALL_DOMAINS.values()
                            for f in spec['features'] + spec.get('flip', [])})
    mouse_by_day = aggregate_to_mouse(baseline_active, MOUSE_ID_COLS + ['day'], raw_features)
    mouse_by_day = compute_domain_scores(mouse_by_day, domain_defs=CORE_DOMAINS)
    wide_by_day = box_paired_wide(mouse_by_day, features=CORE_DOMAIN_FEATURES,
                                   extra_group_cols=('sex', 'day'))

    plot_domain_scores_by_day_grid(
        wide_by_day, CORE_DOMAIN_FEATURES,
        title='BASELINE: ELA vs CTRL, domain scores by day (box-paired)',
    )
