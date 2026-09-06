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

    # or, several domains x both sexes in one reference grid (wraps into extra rows if
    # `features` is long; use pair_sexes=True to put male/female side by side per domain
    # with a shared y-axis scale instead of stacking them in separate rows):
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
from matplotlib.lines import Line2D

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


def _add_shared_legend(fig, sexes, y):
    """Single figure-level legend (CTRL + one ELA entry per sex), placed at figure-fraction
    height `y`, instead of an awkward legend sitting inside one panel."""
    handles = [Line2D([0], [0], color=COLOR_CTRL, marker='o', lw=2.2, markersize=7, label='CTRL')]
    for sex in sexes:
        handles.append(Line2D([0], [0], color=SEX_COLORS.get(sex, '#e34948'), marker='s',
                               lw=2.2, markersize=7, label=f'ELA ({sex})'))
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(0.5, y),
               ncol=len(handles), frameon=False, fontsize=8.5)


def _auto_max_cols(n_items, small_threshold, min_cols):
    """Pick a column count that keeps the grid roughly square once n_items exceeds
    small_threshold, instead of always laying everything out in one long row."""
    if n_items <= small_threshold:
        return n_items
    return max(min_cols, int(np.ceil(np.sqrt(n_items))))


def plot_domain_scores_by_day_grid(wide_by_day, features, sexes=('female', 'male'), day_col='day',
                                    show_individual_boxes=False, annotate_stats=True,
                                    pair_sexes=False, max_cols=None, legend_loc='top',
                                    title=None, save_path=None, show=True, dpi=150):
    """Reference grid of plot_domain_score_by_day() panels, one per feature x sex.

    Two layouts are available:

    - pair_sexes=False (default): one row per sex, one column per feature (as before).
      When `features` is long, the columns wrap into additional row-groups (stacked
      vertically, len(sexes) rows per group) instead of producing one very wide row,
      with `max_cols` (or an automatic sqrt(n)-based choice) controlling the wrap width.
    - pair_sexes=True: requires exactly two sexes. Each feature gets its own male/female
      pair of panels placed side by side, y-axis matched across the pair so the two
      trajectories are directly comparable, with the female panel's y tick labels
      suppressed since it shares the male panel's scale. Feature-pairs wrap into a grid
      the same way when there are many features.

    Args:
        wide_by_day: see plot_domain_score_by_day().
        features: list of domain-score column names.
        sexes: sex order (default female, male). Must have length 2 if pair_sexes=True.
        pair_sexes: if True, plot each feature's two sexes side by side with a shared
            y-axis scale instead of stacking sexes in separate rows.
        max_cols: max number of feature columns (pair_sexes=False) or feature-pair
            blocks (pair_sexes=True) per row before wrapping. Auto-chosen if None.
        legend_loc: 'top' or 'bottom' to place a single shared legend above/below the
            whole grid, or None to omit it.
        save_path: if given, saves the figure there (parent dir created if needed).
        show: whether to call plt.show() (default True; set False for non-interactive use).

    Returns: (fig, axes)
    """
    n_feat = len(features)

    if pair_sexes:
        if len(sexes) != 2:
            raise ValueError('pair_sexes=True requires exactly two sexes')
        sex_left, sex_right = sexes

        ncols_blocks = min(n_feat, max_cols or _auto_max_cols(n_feat, small_threshold=4, min_cols=2))
        nrows_blocks = int(np.ceil(n_feat / ncols_blocks))

        fig, axes = plt.subplots(nrows_blocks, ncols_blocks * 2,
                                  figsize=(2.0 * ncols_blocks * 2, 3.9 * nrows_blocks), squeeze=False)

        for i, feat in enumerate(features):
            row, col_block = divmod(i, ncols_blocks)
            ax_left, ax_right = axes[row, col_block * 2], axes[row, col_block * 2 + 1]

            plot_domain_score_by_day(wide_by_day, feat, sex_left, ax=ax_left, day_col=day_col,
                                      show_individual_boxes=show_individual_boxes,
                                      annotate_stats=annotate_stats)
            plot_domain_score_by_day(wide_by_day, feat, sex_right, ax=ax_right, day_col=day_col,
                                      show_individual_boxes=show_individual_boxes,
                                      annotate_stats=annotate_stats)

            ylo = min(ax_left.get_ylim()[0], ax_right.get_ylim()[0])
            yhi = max(ax_left.get_ylim()[1], ax_right.get_ylim()[1])
            ax_left.set_ylim(ylo, yhi)
            ax_right.set_ylim(ylo, yhi)
            ax_right.tick_params(labelleft=False)

            domain_label = _domain_label(feat)
            ax_left.set_title(f'{domain_label}\n{sex_left}', fontsize=8.5)
            ax_right.set_title(f'{domain_label}\n{sex_right}', fontsize=8.5)
            if col_block == 0:
                ax_left.set_ylabel('z-score', fontsize=9)

        for i in range(n_feat, nrows_blocks * ncols_blocks):
            row, col_block = divmod(i, ncols_blocks)
            axes[row, col_block * 2].axis('off')
            axes[row, col_block * 2 + 1].axis('off')

    else:
        ncols = min(n_feat, max_cols or _auto_max_cols(n_feat, small_threshold=6, min_cols=3))
        n_groups = int(np.ceil(n_feat / ncols))
        nrows = n_groups * len(sexes)

        fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 3.9 * nrows), squeeze=False)

        for group in range(n_groups):
            feats_in_group = features[group * ncols:(group + 1) * ncols]
            for row_offset, sex in enumerate(sexes):
                r = group * len(sexes) + row_offset
                for col, feat in enumerate(feats_in_group):
                    ax = axes[r, col]
                    plot_domain_score_by_day(wide_by_day, feat, sex, ax=ax, day_col=day_col,
                                              show_individual_boxes=show_individual_boxes,
                                              annotate_stats=annotate_stats)
                    ax.set_title(_domain_label(feat), fontsize=9) if row_offset == 0 else ax.set_title('')
                    if col == 0:
                        ax.set_ylabel(f'{sex}\nz-score', fontsize=9)
                for col in range(len(feats_in_group), ncols):
                    axes[r, col].axis('off')

    top_pad, bottom_pad, title_y, legend_y = 0.0, 0.0, 1.02, None
    if legend_loc == 'top':
        top_pad, legend_y = (0.12, 0.935) if title else (0.07, 0.965)
        title_y = 0.99
    elif legend_loc == 'bottom':
        bottom_pad, legend_y = 0.07, 0.015
        if title:
            top_pad, title_y = 0.06, 0.99

    plt.tight_layout(rect=(0, bottom_pad, 1, 1 - top_pad))
    if title:
        fig.suptitle(title, color=COLOR_INK, fontsize=12, fontweight='bold', y=title_y)
    if legend_loc:
        _add_shared_legend(fig, sexes, legend_y)

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
