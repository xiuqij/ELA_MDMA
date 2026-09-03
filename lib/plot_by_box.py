"""
plot_by_box.py - within-box paired dot-line detail plot: one line per box connecting its ELA
value to its CTRL value for one domain score, one panel per (domain, sex) - the individual-box
detail view underneath plot_effect_table.py's forest-plot summary (same paired_effect_table
data source, just showing every box pair instead of collapsing to one dz per panel).

Reusable for any box_paired_wide() output paired along `pair_col` (default 'background': ELA vs
CTRL) - including post-treatment sessions, where an optional `group_col` (e.g. 'treatment')
further splits/colors the box-lines WITHIN each panel, so you can see whether MDMA-dosed boxes
and saline-dosed boxes trace a different ELA-CTRL relationship at the individual-box level, not
just in the aggregate (stats_utils.box_level_delta_table / treatment_interaction.py).

Usage - import into run_analysis.py:

    from plot_by_box import plot_domains_by_box_grid, top_panels_by_effect

    # baseline (step 4.1) - single color per sex, no further split:
    panels = top_panels_by_effect(baseline_effect_tables, n_per_sex=3)
    plot_domains_by_box_grid(wide, panels, title='BASELINE: ELA vs CTRL, box-level detail')

    # post-treatment (step 7.4) - color-split by treatment within each panel:
    plot_domains_by_box_grid(wide_post, panels, group_col='treatment',
        title='POST-INJECTION: ELA vs CTRL, box-level detail, by treatment')

Standalone: `python plot_by_box.py` reproduces run_analysis.py steps 1-4 and shows an example
grid (no save) - smoke test.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from stats_utils import paired_effect_table

SEX_COLORS = {'female': '#e34948', 'male': '#2a78d6'}
GROUP_COLORS_DEFAULT = {'MDMA': '#8e44ad', 'saline': '#898781'}
COLOR_MUTED = '#898781'
COLOR_INK = '#0b0b0b'


def _domain_label(feature):
    return feature.replace('_score', '').replace('_', ' ').capitalize()


def top_panels_by_effect(effect_tables, n_per_sex=3, rank_col='p_fdr'):
    """Pick the top `n_per_sex` domains per sex (ascending `rank_col`, i.e. strongest/most
    significant first) from a {sex: paired_effect_table()} dict - e.g. baseline_effect_tables
    from run_analysis.py step 4. Returns a list of (feature, sex) tuples, in dict order, ready
    to pass as `panels` to plot_domains_by_box_grid()."""
    panels = []
    for sex, res in effect_tables.items():
        top = res.sort_values(rank_col).head(n_per_sex)
        panels += [(f, sex) for f in top['feature']]
    return panels


def plot_domain_by_box(wide, feature, sex=None, ax=None, pair_values=('ELA', 'CTRL'),
                        value_suffix='', group_col=None, group_colors=None, default_color=None,
                        annotate_stats=True, title=None):
    """One panel: paired dot-line plot for one domain, one line per box.

    Args:
        wide: output of aggregate.box_paired_wide() (or an already-filtered subset) - needs
            '<feature><value_suffix>_<pair_values[0]>' / '<feature><value_suffix>_<pair_values[1]>'
            columns. Stats annotation uses stats_utils.paired_effect_table(), which expects the
            default '_ELA'/'_CTRL' suffixes - leave pair_values at its default unless you also
            skip annotate_stats.
        feature: domain-score column name (e.g. 'social_hierarchy_score') - used for the column
            lookup (together with value_suffix) AND for the display label, so keep it the plain
            domain name even when plotting a derived quantity (see value_suffix).
        sex: if given, filters wide to wide.sex==sex first (wide must have a 'sex' column);
            leave None if wide is already sex-filtered or you want sexes pooled in one panel.
        value_suffix: appended to `feature` for the column lookup only (e.g. '__delta' to plot
            box_paired_wide() built on a mouse_level_deltas() table's '<feature>__delta' columns,
            instead of the raw domain-score columns) - the panel label/title still shows the
            plain domain name, since it's still "this domain, ELA vs CTRL", just a different y.
        group_col: optional column in wide (e.g. 'treatment') to color/split box-lines by
            within this panel. If None, all boxes are drawn in one color (default_color, or the
            sex convention color from SEX_COLORS when `sex` is given).
        group_colors: {group_value: color}; falls back to GROUP_COLORS_DEFAULT then
            default_color for any group not covered by either.
        annotate_stats: add box-paired Cohen's dz + p-value to the title - one line overall, or
            one line per group_col value if group_col is given.
        title: panel title; defaults to the domain label (+ sex, if given).

    Returns: ax
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(2.6, 4.3))

    sub = wide[wide['sex'] == sex] if (sex is not None and 'sex' in wide.columns) else wide
    lookup = f'{feature}{value_suffix}'
    col_a, col_b = f'{lookup}_{pair_values[0]}', f'{lookup}_{pair_values[1]}'
    x_a, x_b = 0, 1

    if default_color is None:
        default_color = SEX_COLORS.get(sex, '#2b6a99')
    palette = dict(GROUP_COLORS_DEFAULT)
    if group_colors:
        palette.update(group_colors)

    if group_col is None:
        groups = [(None, sub)]
    else:
        groups = [(g, sub[sub[group_col] == g]) for g in sub[group_col].dropna().unique()]

    for g, gsub in groups:
        color = palette.get(g, default_color) if g is not None else default_color
        a, b = gsub[col_a].values, gsub[col_b].values
        mask = ~(pd.isna(a) | pd.isna(b))
        for ai, bi in zip(a[mask], b[mask]):
            ax.plot([x_a, x_b], [ai, bi], color=color, alpha=0.35, lw=1, zorder=1)
        ax.scatter(np.full(mask.sum(), x_a), a[mask], color=color, s=45, zorder=2,
                   edgecolor='k', linewidth=0.3, label=g)
        ax.scatter(np.full(mask.sum(), x_b), b[mask], facecolor='white', edgecolor=color,
                   s=45, zorder=2, linewidth=1.5)

    ax.set_xticks([x_a, x_b])
    ax.set_xticklabels(pair_values)
    ax.set_xlim(-0.5, 1.5)
    ax.axhline(0, color=COLOR_MUTED, lw=0.5, ls=':')

    panel_title = title or (_domain_label(feature) + (f'\n({sex})' if sex else ''))
    if annotate_stats:
        stat_lines = []
        if group_col is None:
            res = paired_effect_table(sub, [lookup])
            if len(res):
                dz, p = res.loc[0, 'cohen_dz'], res.loc[0, 'p_ttest']
                star = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
                stat_lines.append(f'dz={dz:+.2f}, p={p:.3f} {star}')
        else:
            for g, gsub in groups:
                res = paired_effect_table(gsub, [lookup])
                if len(res):
                    dz, p = res.loc[0, 'cohen_dz'], res.loc[0, 'p_ttest']
                    stat_lines.append(f'{g}: dz={dz:+.2f}, p={p:.3f}')
        if stat_lines:
            panel_title += '\n' + '\n'.join(stat_lines)
    ax.set_title(panel_title, fontsize=9)
    sns.despine(ax=ax)
    return ax


def plot_domains_by_box_grid(wide, panels, group_col=None, group_colors=None, value_suffix='',
                              annotate_stats=True, title=None,
                              ylabel='z-scored composite (per-box mean)',
                              save_path=None, show=True, dpi=150):
    """Row of panels via plot_domain_by_box(), one per (feature, sex) pair in `panels`.

    Args:
        wide: box_paired_wide() output covering all sexes/groups needed by `panels`.
        panels: list of (feature, sex) tuples - e.g. from top_panels_by_effect(), or a curated
            list built by hand.
        group_col / group_colors: passed through to every panel - see plot_domain_by_box().
        value_suffix: passed through to every panel - see plot_domain_by_box() (e.g. '__delta').
        save_path: if given, saves the figure there (parent dir created if needed).

    Returns: (fig, axes)
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.7 * n, 4.4), squeeze=False)
    axes = axes[0]

    for ax, (feat, sex) in zip(axes, panels):
        plot_domain_by_box(wide, feat, sex=sex, ax=ax, group_col=group_col,
                            group_colors=group_colors, value_suffix=value_suffix,
                            annotate_stats=annotate_stats)
        if ax is axes[0] and ylabel:
            ax.set_ylabel(ylabel)

    if group_col is not None:
        handles, labels = axes[0].get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l and l not in seen:
                seen[l] = h
        if seen:
            fig.legend(seen.values(), seen.keys(), loc='lower center', ncol=len(seen),
                       frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.04))

    if title:
        fig.suptitle(title, color=COLOR_INK, fontsize=12, fontweight='bold', y=1.06)
    plt.tight_layout(rect=(0, 0.05, 1, 1) if group_col is not None else (0, 0, 1, 1))

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes


#%%
# ============================================================================
# STANDALONE SMOKE TEST - reproduces run_analysis.py steps 1-4 and calls
# plot_domains_by_box_grid() directly, without saving. Only runs when this file is executed on
# its own (`python plot_by_box.py`), not when imported.
# ============================================================================
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')

    from prep import load_data, filter_data
    from aggregate import aggregate_to_mouse, box_paired_wide
    from domain_scores import compute_domain_scores, ALL_DOMAINS, ALL_DOMAIN_FEATURES

    MOUSE_ID_COLS = ['mouse_ID', 'sex', 'age', 'background', 'treatment', 'box_ID']
    DATA_PATH = '../behavior_dataset/final/QC_table'

    df = load_data(female_path=f'{DATA_PATH}/female_12h_filtered.csv',
                    male_path=f'{DATA_PATH}/male_12h_filtered.csv')
    baseline_active = filter_data(df, phase='active', time_point='baseline')

    raw_features = sorted({f for spec in ALL_DOMAINS.values()
                            for f in spec['features'] + spec.get('flip', [])})
    mouse_baseline = aggregate_to_mouse(baseline_active, MOUSE_ID_COLS, raw_features)
    mouse_baseline = compute_domain_scores(mouse_baseline, domain_defs=ALL_DOMAINS)
    wide = box_paired_wide(mouse_baseline, features=ALL_DOMAIN_FEATURES)

    effect_tables = {sex: paired_effect_table(wide[wide.sex == sex], ALL_DOMAIN_FEATURES)
                      for sex in ['female', 'male']}
    panels = top_panels_by_effect(effect_tables, n_per_sex=3)

    plot_domains_by_box_grid(wide, panels, title='BASELINE: ELA vs CTRL, box-level detail (smoke test)')
