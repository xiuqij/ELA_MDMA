"""
plot_effect_table.py - forest plot for one or more paired_effect_table() results.

Reusable plotting function meant to be imported wherever you already have
paired_effect_table() output (e.g. one table per sex) and want to visualize it: one row
per feature, x = effect size (default Cohen's dz), colored by direction, filled markers
for FDR-significant hits, one panel per entry in `results`.

Usage - import into run_analysis.py (or any other script) right after building the
per-sex effect tables in step 4:

    from plot_effect_table import plot_effect_table

    results = {}
    for sex in ['female', 'male']:
        sub = wide[wide.sex == sex]
        results[sex] = paired_effect_table(sub, ALL_DOMAIN_FEATURES)

    plot_effect_table(
        results,
        title='BASELINE: ELA vs CTRL, domain scores (box-paired)',
        save_path='baseline_effect_table.png',   # None to skip saving
    )

Standalone: `python plot_effect_table.py` reproduces run_analysis.py steps 1-4 on its own
and shows the same plot (no save) as a smoke test / example.
"""
#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from domain_scores import ALL_DOMAIN_FEATURES

# Palette (dataviz skill diverging pair: blue <-> red, neutral gray midpoint).
# POS = effect_col >= 0 (i.e. ELA > CTRL with the default cohen_dz sign convention),
# NEG = effect_col < 0 (ELA < CTRL).
COLOR_POS = '#e34948'   # red
COLOR_NEG = '#2a78d6'   # blue
COLOR_MUTED = '#898781'
COLOR_INK = '#0b0b0b'

# Fixed top-to-bottom display order for the forest plot, so the same domain sits on the
# same row across panels and panels are easy to compare side by side. Edit this list to
# reorder/subset the domains shown - defaults to domain_scores.ALL_DOMAIN_FEATURES (CORE_
# DOMAINS then SUPPLEMENTARY_DOMAINS, in their definition order there).
FEATURE_ORDER = list(ALL_DOMAIN_FEATURES)


def plot_effect_table(results, effect_col='cohen_dz', sig_col='p_fdr', sig_thresh=0.05,
                       feature_order=None,
                       pos_label='ELA > CTRL', neg_label='ELA < CTRL',
                       xlabel="Cohen's dz (ELA - CTRL, box-paired)",
                       title=None, save_path=None, show=True, dpi=150):
    """Forest plot of an effect-size table (or one per panel), in a fixed row order shared
    across all panels so the same domain lines up on the same row in every panel.

    Args:
        results: dict of {panel_label: DataFrame}, e.g. {'female': res_f, 'male': res_m}.
            Each DataFrame needs a 'feature' column, `effect_col`, and `sig_col` - i.e. the
            direct output of stats_utils.paired_effect_table(). A single DataFrame is also
            accepted and treated as a one-panel dict.
        effect_col: column plotted on the x-axis (default 'cohen_dz').
        sig_col: column used to decide filled-vs-hollow markers (default 'p_fdr').
        sig_thresh: significance threshold on `sig_col` (default 0.05).
        feature_order: top-to-bottom list of feature names shared by all panels; defaults to
            the module-level FEATURE_ORDER (edit that list to change the default). Any
            feature in a panel's table but not in this list is dropped from that panel; any
            entry in this list absent from a given panel's table is skipped for that panel.
        pos_label / neg_label: legend text for effect_col >= 0 / < 0.
        xlabel: x-axis label, shared across panels.
        title: figure suptitle (optional).
        save_path: if given, saves the figure there (parent dir created if needed).
        show: whether to call plt.show() (default True; set False for non-interactive use).
        dpi: resolution used when saving.

    Returns:
        (fig, axes)
    """
    if hasattr(results, 'columns'):  # a bare DataFrame, not a dict
        results = {'': results}

    order = feature_order if feature_order is not None else FEATURE_ORDER

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 8), sharex=True, squeeze=False)
    axes = axes[0]

    for ax, (panel_label, res) in zip(axes, results.items()):
        panel_order = [f for f in order if f in set(res['feature'])]
        res = res.set_index('feature').loc[panel_order].reset_index()
        y = np.arange(len(res))
        colors = np.where(res[effect_col] >= 0, COLOR_POS, COLOR_NEG)
        sig = res[sig_col] < sig_thresh

        ax.hlines(y, 0, res[effect_col], color=colors, linewidth=1.5, zorder=1)
        ax.scatter(res.loc[sig, effect_col], y[sig], color=colors[sig], s=60,
                   edgecolor=colors[sig], linewidth=1, zorder=2)
        ax.scatter(res.loc[~sig, effect_col], y[~sig], facecolor='white',
                   edgecolor=colors[~sig], linewidth=1.5, s=60, zorder=2)

        ax.axvline(0, color=COLOR_MUTED, linewidth=1, linestyle='--', zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels(res['feature'], fontsize=8)
        ax.invert_yaxis()  # first entry in `order` on top
        ax.set_xlabel(xlabel)
        subtitle = panel_label
        if 'n_pairs' in res.columns and len(res):
            subtitle = f'{panel_label} (n={res["n_pairs"].iloc[0]} box pairs)'
        ax.set_title(subtitle)
        ax.tick_params(axis='y', length=0)
        sns.despine(ax=ax, left=True)
        ax.grid(axis='x', color=COLOR_MUTED, alpha=0.25, linewidth=0.5)

    handles = [
        plt.Line2D([0], [0], marker='o', color=COLOR_POS, linestyle='', markersize=8,
                   label=f'{pos_label}, {sig_col} < {sig_thresh}'),
        plt.Line2D([0], [0], marker='o', color=COLOR_NEG, linestyle='', markersize=8,
                   label=f'{neg_label}, {sig_col} < {sig_thresh}'),
        plt.Line2D([0], [0], marker='o', markerfacecolor='white', markeredgecolor=COLOR_MUTED,
                   linestyle='', markersize=8, label=f'n.s. ({sig_col} >= {sig_thresh})'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    if title:
        fig.suptitle(title, color=COLOR_INK)
    plt.tight_layout(rect=(0, 0.03, 1, 1))

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()

    return fig, axes


#%%
# ============================================================================
# STANDALONE SMOKE TEST - reproduces run_analysis.py steps 1-4 and calls
# plot_effect_table() directly, without saving. Only runs when this file is executed
# on its own (`python plot_effect_table.py`), not when imported.
# ============================================================================
if __name__ == '__main__':
    import sys
    sys.path.insert(0, 'lib')

    from prep import load_data, filter_data
    from aggregate import aggregate_to_mouse, box_paired_wide
    from domain_scores import compute_domain_scores, ALL_DOMAINS
    from stats_utils import paired_effect_table

    MOUSE_ID_COLS = ['mouse_ID', 'sex', 'age', 'background', 'treatment', 'box_ID']
    DATA_PATH = '/Users/xiuqi.ji/Library/CloudStorage/OneDrive-KarolinskaInstitutet/MDMA_ELA/social_box_2026/github_repo/behavior_dataset/final/QC_table'

    df = load_data(female_path=f'{DATA_PATH}/female_12h_filtered.csv',
                    male_path=f'{DATA_PATH}/male_12h_filtered.csv')
    baseline_active_day23 = filter_data(df, phase='active', time_point='baseline', days=[2, 3])

    raw_features = sorted({f for spec in ALL_DOMAINS.values()
                            for f in spec['features'] + spec.get('flip', [])})
    mouse_baseline_day23 = aggregate_to_mouse(baseline_active_day23, MOUSE_ID_COLS, raw_features)
    mouse_baseline_day23 = compute_domain_scores(mouse_baseline_day23, domain_defs=ALL_DOMAINS)
    wide = box_paired_wide(mouse_baseline_day23, features=ALL_DOMAIN_FEATURES)

    results = {sex: paired_effect_table(wide[wide.sex == sex], ALL_DOMAIN_FEATURES)
               for sex in ['female', 'male']}

    plot_effect_table(results, title='BASELINE: ELA vs CTRL, domain scores (box-paired)')
