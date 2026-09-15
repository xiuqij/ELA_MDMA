"""
treatment_interaction.py - does MDMA affect ELA and CTRL animals differently, and what does
that difference look like?

Five angles, each a compute_* (tidy DataFrame) + plot_* (visualization) pair, plus a raw-data
EDA plot (plot_deltas_by_condition_grid) meant to be looked at BEFORE any of the others:

  0. Raw deltas by condition - plain box+strip plot of the per-mouse delta in each of the 4
     background x treatment conditions (CTRL_saline, CTRL_MDMA, ELA_saline, ELA_MDMA), one panel
     per domain x sex, no model fit involved - a sanity check on the data the mixed model below
     is about to summarize.
       plot: plot_deltas_by_condition_grid()
  0.5. General MDMA effect (background POOLED) - does MDMA change this domain at all, regardless
     of ELA/CTRL? The 2 ELA + 2 CTRL mice in a box are averaged together (same box-level,
     unpaired MDMA-vs-saline test as stats_utils.box_level_delta_table, already used as plain
     printed text in run_analysis.py step 7) - here given a tidy table + forest-plot summary,
     plus the day-level and time-window progression of that same pooled effect.
       compute: compute_general_mdma_effect()      plot: plot_general_mdma_volcano()
                                                     (all-domain overview, effect size vs
                                                     significance - a scatter, not a forest plot,
                                                     so it doesn't read as another instance of the
                                                     paired ELA-vs-CTRL forest plots elsewhere;
                                                     plot_general_mdma_forest() is the earlier,
                                                     same-style-as-those alternative, kept for
                                                     reuse but no longer the default)
       compute: compute_pooled_progression()        plot: plot_pooled_treatment_progression_grid()
                                                     (day-level OR time-window progression, same
                                                     background-pooled MDMA-vs-saline test repeated
                                                     per day / per window - pass window_col='day' on
                                                     the 12h table, or 'time_window' on a finer one)
  1. Background x treatment interaction - does ELA blunt/amplify the drug effect, domain by
     domain? (this is the generalized, reusable version of the old fig5_interaction.py, which
     hardcoded 4 panels and read from two stale intermediate CSVs that no longer exist)
       compute: run_domain_mixedmodels()        plot: plot_bg_x_treatment_interaction()
                                                 plot: plot_mixedmodel_forest() (all-domain overview)
  2. Normalization - does MDMA move ELA animals TOWARD the CTRL baseline level (potential
     therapeutic direction), beyond whatever shift saline-dosed boxes show on retest alone?
       compute: compute_normalization_table()   plot: plot_normalization()
       compute: compute_normalization_gaps()    plot: plot_normalization_gap_grid()
         (per-box gap distribution, one panel per domain, underlying the dz summary above)
  3. Time-course - is the drug effect front-loaded (early post-injection hours) or does it
     accumulate/persist across the whole session? Needs a finer-resolution QC_table file.
       compute: compute_window_deltas()         plot: plot_delta_by_window_grid()
         (background SPLIT - CTRL/ELA drawn as separate lines within MDMA-dosed boxes; see 0.5
         above for the background-POOLED version of this same window_deltas table)

Design notes carried over from prep.py / aggregate.py / README_preliminary_analysis.md:
  - `treatment` (MDMA/saline) is a BOX-level factor (all mice in a box get the same substance).
  - `background` (ELA/CTRL) is balanced 2+2 WITHIN every box, independent of treatment -> the
    box-paired ELA-vs-CTRL comparison (paired_effect_table) is valid at any session/treatment
    combination, not just at baseline.
  - `time_point` (baseline/MDMA) labels the SESSION (pre- vs post-injection), recorded for every
    box regardless of what it received - so time_point=='MDMA' rows in a saline-dosed box are
    the post-*saline*-injection session, used here as the non-specific retest/practice control.

Usage - import into run_analysis.py step 7 right after building `deltas` and `mouse_sess`:

    from treatment_interaction import (
        plot_deltas_by_condition_grid,
        compute_general_mdma_effect, plot_general_mdma_volcano, plot_general_mdma_forest,
        compute_pooled_progression, plot_pooled_treatment_progression_grid,
        run_domain_mixedmodels, plot_bg_x_treatment_interaction, plot_mixedmodel_forest,
        compute_normalization_table, plot_normalization,
        compute_normalization_gaps, plot_normalization_gap_grid,
        compute_window_deltas, plot_delta_by_window_grid,
    )
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from statsmodels.stats.multitest import multipletests

from aggregate import aggregate_to_mouse, box_paired_wide, mouse_level_deltas
from domain_scores import compute_domain_scores
from stats_utils import paired_effect_table, run_mixedlm, box_level_delta_table

# Palette conventions consistent with plot_effect_table.py / plot_domain_score_by_day.py
# (dataviz skill diverging pair + the sex/background convention already used there).
COLOR_CTRL = '#898781'
COLOR_MUTED = '#898781'
COLOR_INK = '#0b0b0b'
SEX_ELA_COLORS = {'female': '#e34948', 'male': '#2a78d6'}
COLOR_MDMA_GROUP = '#8e44ad'
COLOR_SALINE_GROUP = '#898781'
COLOR_SHRINK = '#2f7d5c'   # gap narrows (normalization) - green
COLOR_WIDEN = '#a8583f'    # gap widens (anti-normalization) - rust
COLOR_ACUTE = '#c9762d'    # Day 1 (single-dose day) - amber
COLOR_SUSTAINED = '#2f7f8f'  # Days 2-3 (drug-free follow-up) - teal


def _domain_label(feature):
    return feature.replace('_score', '').replace('_', ' ').capitalize()


def _legend_layout(legend_loc, title):
    """tight_layout padding + suptitle/legend y-positions for a shared figure-level legend
    placed 'top' or 'bottom' relative to the grid, instead of inside any one subplot - same
    convention as temporal_dynamics.py's identically-named helper."""
    top_pad, bottom_pad, title_y, legend_y = 0.0, 0.0, 1.02, None
    if legend_loc == 'top':
        top_pad, legend_y = (0.12, 0.935) if title else (0.07, 0.965)
        title_y = 0.99
    elif legend_loc == 'bottom':
        bottom_pad, legend_y = 0.07, 0.015
        if title:
            top_pad, title_y = 0.06, 0.99
    return top_pad, bottom_pad, title_y, legend_y


# ============================================================================
# 0. RAW DELTAS BY CONDITION (EDA, no model fit - look at this before 1.)
# ============================================================================
CONDITION_ORDER = ['CTRL_saline', 'CTRL_MDMA', 'ELA_saline', 'ELA_MDMA']


def _condition_palette(sex):
    ela_color = SEX_ELA_COLORS.get(sex, '#e34948')
    return {'CTRL_saline': COLOR_CTRL, 'CTRL_MDMA': COLOR_CTRL,
            'ELA_saline': ela_color, 'ELA_MDMA': ela_color}


def plot_deltas_by_condition_grid(deltas, features, sexes=('female', 'male'),
                                   match_ylim_by_domain=False, ref_condition='CTRL_saline',
                                   title=None, save_path=None, show=True, dpi=150):
    """Raw-data look at the per-mouse delta (post-injection - baseline) in each of the 4
    background x treatment conditions, BEFORE fitting the delta ~ background * treatment mixed
    model in run_domain_mixedmodels() below - a sanity check on what that model is about to
    summarize (outliers, obviously non-normal spread, near-empty cells, etc.).

    One panel per domain (columns) x sex (rows): box + individual jittered points, x-axis =
    CTRL_saline, CTRL_MDMA, ELA_saline, ELA_MDMA. Saline-dosed boxes (whose "post-injection"
    session is really just a retest, not a drug challenge - see module docstring) are drawn at
    reduced alpha so the eye lands on the two MDMA columns first.

    Args:
        match_ylim_by_domain: if True, the two sex rows sharing the same domain (column) are
            given a common y-axis range - the union of their individual auto-scaled ranges - so
            female and male are directly comparable at a glance instead of each panel picking
            its own scale. No effect when only one sex is plotted. Same convention as
            plot_by_box.py's plot_domains_by_box_grid(match_ylim_by_domain=...).
        ref_condition: if given (default 'CTRL_saline'), draws a dashed reference line at that
            condition's mean delta in every panel, so the other three conditions are read
            relative to the untouched/no-drug baseline group rather than only against zero.
            Pass None to omit it.

    Input: deltas - output of aggregate.mouse_level_deltas() (needs 'background', 'treatment',
    'sex' plus '<feature>__delta' for each feature in `features`).

    Returns: (fig, axes)
    """
    d = deltas.copy()
    d['condition'] = d['background'] + '_' + d['treatment']

    fig, axes = plt.subplots(len(sexes), len(features),
                              figsize=(2.6 * len(features), 4.0 * len(sexes)), squeeze=False)

    for row, sex in enumerate(sexes):
        pal = _condition_palette(sex)
        sub_sex = d[d.sex == sex]
        for col, feat in enumerate(features):
            ax = axes[row, col]
            y_col = f'{feat}__delta'
            sub = sub_sex[['condition', y_col]].dropna()

            sns.boxplot(data=sub, x='condition', y=y_col, hue='condition', order=CONDITION_ORDER,
                        hue_order=CONDITION_ORDER, palette=pal, showfliers=False, width=0.6,
                        dodge=False, legend=False, ax=ax)
            for patch, cond in zip(ax.patches, CONDITION_ORDER):
                patch.set_alpha(0.35 if cond.endswith('saline') else 0.6)
            sns.stripplot(data=sub, x='condition', y=y_col, hue='condition', order=CONDITION_ORDER,
                          hue_order=CONDITION_ORDER, palette=pal, size=4, jitter=0.15,
                          linewidth=0.3, edgecolor='white', dodge=False, legend=False, ax=ax)
            for coll, cond in zip(ax.collections, CONDITION_ORDER):
                coll.set_alpha(0.4 if cond.endswith('saline') else 0.85)

            ax.axhline(0, color=COLOR_MUTED, lw=0.5, ls=':', zorder=0)
            if ref_condition:
                ref_vals = sub.loc[sub.condition == ref_condition, y_col]
                if len(ref_vals):
                    ax.axhline(ref_vals.mean(), color=COLOR_INK, lw=1.1, ls='--', alpha=0.6,
                               zorder=1, label=f'{ref_condition} mean')
            ax.set_xticks(range(len(CONDITION_ORDER)))
            ax.set_xticklabels(['CTRL\nsaline', 'CTRL\nMDMA', 'ELA\nsaline', 'ELA\nMDMA'],
                               fontsize=7.5)
            ax.set_xlabel('')
            ax.set_title(_domain_label(feat) if row == 0 else '', fontsize=9.5)
            ax.set_ylabel(f'{sex}\nΔ (post − baseline)\nz-scored composite' if col == 0 else '',
                          fontsize=9)
            sns.despine(ax=ax)

    if match_ylim_by_domain and len(sexes) > 1:
        for col in range(len(features)):
            col_axes = axes[:, col]
            lo = min(a.get_ylim()[0] for a in col_axes)
            hi = max(a.get_ylim()[1] for a in col_axes)
            for a in col_axes:
                a.set_ylim(lo, hi)

    if ref_condition:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center', ncol=1, frameon=False, fontsize=9,
                       bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title or 'Treatment deltas by condition (raw data, before model fit)',
                 color=COLOR_INK, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(rect=(0, 0.02, 1, 1) if ref_condition else (0, 0, 1, 1))

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes


# ============================================================================
# 0.5 GENERAL MDMA EFFECT (background POOLED) - does MDMA change behavior at all, regardless of
# ELA/CTRL? Companion to section 1 below, which asks the different question of whether ELA
# changes the SIZE of that effect.
# ============================================================================
def compute_general_mdma_effect(deltas, features, sexes=('female', 'male'), box_col='box_ID'):
    """Box-level MDMA-vs-saline test on the post-baseline delta, background POOLED within box
    (the 2 ELA + 2 CTRL mice in a box are averaged together first, same as
    stats_utils.box_level_delta_table's default) - "does MDMA change this domain at all,
    regardless of ELA/CTRL?" Thin tidy-table wrapper so the result feeds a forest plot the same
    way run_domain_mixedmodels() feeds plot_mixedmodel_forest().

    Input: deltas - output of aggregate.mouse_level_deltas() (needs 'box_ID', 'sex' plus
    '<feature>__delta' for each feature in `features`).

    Returns tidy long table: domain, sex, n_A, n_B, mean_delta_A, mean_delta_B, hedges_g, t, p,
    p_mannwhitney, p_fdr (A=MDMA, B=saline; p_fdr corrects across domains within each sex).
    """
    rows = []
    for sex in sexes:
        sub = deltas[deltas.sex == sex]
        res = box_level_delta_table(sub, features, box_col=box_col)
        if len(res):
            res = res.rename(columns={'feature': 'domain', 'p_ttest': 'p'})
            res['sex'] = sex
            rows.append(res)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_general_mdma_forest(general_effect, feature_order=None, sexes=('female', 'male'),
                              sig_col='p_fdr', sig_thresh=0.05, title=None, save_path=None,
                              show=True, dpi=150):
    """All-domain forest plot of the general (background-pooled) MDMA vs saline effect on the
    post-baseline delta (Hedges' g, box-level, unpaired) - "does MDMA change this domain at all,
    regardless of ELA/CTRL background?" Same visual language as plot_mixedmodel_forest(), which
    answers the different question of whether ELA modulates that effect.

    Args:
        general_effect: output of compute_general_mdma_effect().
        sig_col: 'p' (nominal) or 'p_fdr' - which column decides filled-vs-hollow markers.

    Returns: (fig, axes)
    """
    order = feature_order if feature_order is not None else list(dict.fromkeys(general_effect['domain']))

    fig, axes = plt.subplots(1, len(sexes),
                              figsize=(6.5 * len(sexes), max(4, 0.45 * len(order))),
                              sharex=True, squeeze=False)
    axes = axes[0]

    for ax, sex in zip(axes, sexes):
        res = (general_effect[general_effect.sex == sex].set_index('domain').reindex(order)
               .dropna(subset=['hedges_g']).reset_index())
        y = np.arange(len(res))
        colors = np.where(res['hedges_g'] >= 0, '#e34948', '#2a78d6')
        sig = res[sig_col] < sig_thresh

        ax.hlines(y, 0, res['hedges_g'], color=colors, linewidth=1.5, zorder=1)
        ax.scatter(res.loc[sig, 'hedges_g'], y[sig], color=colors[sig], s=70, zorder=2)
        ax.scatter(res.loc[~sig, 'hedges_g'], y[~sig], facecolor='white', edgecolor=colors[~sig],
                   linewidth=1.5, s=70, zorder=2)
        ax.axvline(0, color=COLOR_MUTED, linewidth=1, linestyle='--', zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([_domain_label(f) for f in res['domain']], fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel("Hedges' g (MDMA − saline)", fontsize=11)
        n_a = res["n_A"].iloc[0] if len(res) else "-"
        n_b = res["n_B"].iloc[0] if len(res) else "-"
        ax.set_title(f'{sex} (n={n_a} vs {n_b} boxes)', fontsize=13)
        ax.tick_params(axis='y', length=0)
        sns.despine(ax=ax, left=True)
        ax.grid(axis='x', color=COLOR_MUTED, alpha=0.25, linewidth=0.5)

    handles = [
        plt.Line2D([0], [0], marker='o', color='#e34948', linestyle='', markersize=10,
                   label=f'g >= 0, {sig_col} < {sig_thresh}'),
        plt.Line2D([0], [0], marker='o', color='#2a78d6', linestyle='', markersize=10,
                   label=f'g < 0, {sig_col} < {sig_thresh}'),
        plt.Line2D([0], [0], marker='o', markerfacecolor='white', markeredgecolor=COLOR_MUTED,
                   linestyle='', markersize=10, label=f'n.s. ({sig_col} >= {sig_thresh})'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, -0.03))
    if title:
        fig.suptitle(title, color=COLOR_INK, fontsize=14)
    plt.tight_layout(rect=(0, 0.03, 1, 1))

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes


def plot_general_mdma_volcano(general_effect, sexes=('female', 'male'), sig_col='p_fdr',
                               sig_thresh=0.05, label_all=True, title=None, save_path=None,
                               show=True, dpi=150):
    """Volcano-style overview of the general (background-pooled) MDMA vs saline effect: effect
    size (x, Hedges' g) vs significance (y, -log10 nominal p), one point per domain, one panel
    per sex - both panels share the same x/y scale so the two sexes are directly comparable at a
    glance, not just internally consistent. Deliberately a DIFFERENT chart grammar from
    plot_mixedmodel_forest() / plot_general_mdma_forest() / plot_effect_table() (all
    lollipop-from-zero forest plots used elsewhere for the paired ELA-vs-CTRL comparisons) so
    this whole-session "does MDMA change anything at all" screen doesn't read as another
    instance of that same paired comparison - same red/blue-by-direction and
    filled/hollow-by-significance conventions, just a scatter against a significance axis
    instead of bars from a shared zero baseline.

    Args:
        general_effect: output of compute_general_mdma_effect().
        sig_col: 'p' (nominal) or 'p_fdr' - which column decides filled-vs-hollow markers.
        label_all: label every domain (default) - point positions are shared/comparable across
            panels, so an unlabelled dot in one panel can't be looked up against the other; set
            False to only label domains with nominal p < .2 if a sparser panel is wanted instead.
            Uses the `adjustText` package (if installed) to spread labels apart automatically;
            falls back to a simpler fixed-offset placement otherwise (may overlap when many
            domains cluster near p~1 in the same corner).

    Returns: (fig, axes)
    """
    try:
        from adjustText import adjust_text
        have_adjust_text = True
    except ImportError:
        have_adjust_text = False

    general_effect = general_effect.copy()
    general_effect['neglog10p'] = -np.log10(general_effect['p'].clip(lower=1e-12))
    # shared x/y range across both sex panels, computed once from the pooled data (+ margin)
    x_all, y_all = general_effect['hedges_g'], general_effect['neglog10p']
    x_pad = 0.15 * (x_all.max() - x_all.min())
    xlim = (x_all.min() - x_pad, x_all.max() + x_pad)
    ylim = (-0.15, y_all.max() * 1.18)

    fig, axes = plt.subplots(1, len(sexes), figsize=(6.8 * len(sexes), 5.4), squeeze=False,
                              gridspec_kw={'wspace': 0.3})
    axes = axes[0]

    for ax, sex in zip(axes, sexes):
        res = general_effect[general_effect.sex == sex]
        colors = np.where(res['hedges_g'] >= 0, '#e34948', '#2a78d6')
        sig = (res[sig_col] < sig_thresh).values

        ax.scatter(res.loc[sig, 'hedges_g'], res.loc[sig, 'neglog10p'], color=colors[sig],
                   s=95, zorder=3, edgecolor='white', linewidth=0.9)
        ax.scatter(res.loc[~sig, 'hedges_g'], res.loc[~sig, 'neglog10p'], facecolor='white',
                   edgecolor=colors[~sig], linewidth=1.5, s=70, zorder=2)

        ax.axvline(0, color=COLOR_MUTED, lw=1, ls='--', zorder=0)
        ax.axhline(-np.log10(0.05), color=COLOR_MUTED, lw=0.8, ls=':', zorder=0)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        to_label = res if label_all else res[res['p'] < 0.2]
        texts = []
        x_mid = sum(xlim) / 2
        for _, row in to_label.iterrows():
            if have_adjust_text:
                texts.append(ax.text(row['hedges_g'], row['neglog10p'],
                                      _domain_label(row['domain']), fontsize=9, va='center',
                                      ha='center', color=COLOR_INK))
            else:
                # label points TOWARD the axes center (not by the sign of g) so a point near
                # either edge never has its label pushed straight into that edge and clipped
                toward_right = row['hedges_g'] < x_mid
                xoff = 8 if toward_right else -8
                ha = 'left' if toward_right else 'right'
                ax.annotate(_domain_label(row['domain']), (row['hedges_g'], row['neglog10p']),
                            xytext=(xoff, 0), textcoords='offset points', fontsize=8.5,
                            va='center', ha=ha, color=COLOR_INK, clip_on=True)
        if have_adjust_text and texts:
            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle='-', color=COLOR_MUTED, lw=0.6, alpha=0.7),
                        expand=(1.3, 1.6), force_text=(0.3, 0.4))

        ax.set_xlabel("Hedges' g (MDMA − saline)", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel('significance  →\n-log10(nominal p)', fontsize=10.5)
        n_a = res['n_A'].iloc[0] if len(res) else '-'
        n_b = res['n_B'].iloc[0] if len(res) else '-'
        ax.set_title(f'{sex} (n={n_a} vs {n_b} boxes)', fontsize=13)
        sns.despine(ax=ax)
        ax.grid(alpha=0.2, color=COLOR_MUTED, linewidth=0.5)

    handles = [
        plt.Line2D([0], [0], marker='o', color='#e34948', linestyle='', markersize=10,
                   label=f'g >= 0, {sig_col} < {sig_thresh}'),
        plt.Line2D([0], [0], marker='o', color='#2a78d6', linestyle='', markersize=10,
                   label=f'g < 0, {sig_col} < {sig_thresh}'),
        plt.Line2D([0], [0], marker='o', markerfacecolor='white', markeredgecolor=COLOR_MUTED,
                   linestyle='', markersize=10, label=f'n.s. ({sig_col} >= {sig_thresh})'),
        plt.Line2D([0], [0], color=COLOR_MUTED, lw=0.8, ls=':', label='nominal p = .05'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.05))
    if title:
        fig.suptitle(title, color=COLOR_INK, fontsize=14)
    plt.tight_layout(rect=(0, 0.05, 1, 1))

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes


def compute_pooled_progression(window_deltas, features, window_col='day', sexes=('female', 'male'),
                                box_col='box_ID'):
    """Per (domain, sex, window) box-level MDMA-vs-saline test, background POOLED - the
    per-window (or per-day) significance underlying plot_pooled_treatment_progression()'s '*'
    markers. Nominal p only (no FDR across windows - these are exploratory per-timepoint
    markers, not a replacement for the whole-session compute_general_mdma_effect() test above,
    which is FDR-corrected).

    window_col: a single column name (e.g. 'day', or 'time_window' if `window_deltas` has no
    'day' column) or a list e.g. ['day', 'time_window'] to test each treatment day's
    within-session windows separately, instead of pooling all post-treatment days into one
    per-clock-hour number - see compute_window_deltas()'s id_cols for how to build a
    'day'-resolved `window_deltas` in the first place.

    Input: window_deltas - any mouse_level_deltas() table with the `window_col` column(s)
    retained as extra grouping columns, e.g. compute_window_deltas(df, ..., window_col='day') on
    the 12h-resolution table for day-level progression, or
    compute_window_deltas(df_fine, ..., id_cols=MOUSE_ID_COLS + ['day']) at finer resolution for
    within-session-per-day progression.

    Returns tidy long table: domain, sex, <window_col column(s)>, hedges_g, p, n_A, n_B.
    """
    cols = [window_col] if isinstance(window_col, str) else list(window_col)
    rows = []
    for sex in sexes:
        sub_sex = window_deltas[window_deltas.sex == sex]
        for combo in _ordered_combos(sub_sex, cols):
            sub_w = sub_sex[_combo_mask(sub_sex, cols, combo)]
            res = box_level_delta_table(sub_w, features, box_col=box_col)
            if len(res):
                res = res.rename(columns={'feature': 'domain', 'p_ttest': 'p'})
                res['sex'] = sex
                for c, v in zip(cols, combo):
                    res[c] = v
                rows.append(res)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_pooled_treatment_progression(window_deltas, feature, sex, ax=None, window_col='day',
                                       stats_table=None, box_col='box_ID', show_individual=True,
                                       active_start_hour=12):
    """One domain's MDMA-vs-baseline delta across `window_col` (e.g. 'day', 'time_window', or
    ['day', 'time_window'] for a day-resolved within-session time-course), background POOLED -
    mean+/-SEM per treatment group only (MDMA solid purple vs saline dashed grey), individual
    mice shown as faint jittered points. The background-pooled, general-effect companion to
    plot_delta_by_window() (which instead splits MDMA-dosed boxes by background and draws saline
    only as a faint reference line) - same window_col conventions as that function.

    stats_table: output of compute_pooled_progression() (with a matching window_col) - if given,
    marks windows where the box-level MDMA-vs-saline test reaches nominal p<.05 with an asterisk
    above that point.

    Returns: ax
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3.8, 4.3))

    cols = [window_col] if isinstance(window_col, str) else list(window_col)
    col = f'{feature}__delta'
    sub = window_deltas[window_deltas.sex == sex][['treatment', *cols, col]].dropna()
    combos = _ordered_combos(sub, cols)
    x = np.arange(len(combos))

    rng = np.random.default_rng(0)
    group_means = {}
    for treat, style, xoff in [('saline', dict(color=COLOR_SALINE_GROUP, marker='o', ls='--'), -0.08),
                                ('MDMA', dict(color=COLOR_MDMA_GROUP, marker='o', ls='-'), 0.08)]:
        g = sub[sub.treatment == treat]
        y, e = [], []
        for xi, c in enumerate(combos):
            pts = g.loc[_combo_mask(g, cols, c), col]
            y.append(pts.mean())
            e.append(pts.sem())
            if show_individual and len(pts):
                jitter = rng.uniform(-0.06, 0.06, size=len(pts))
                ax.scatter(np.full(len(pts), xi) + xoff + jitter, pts, color=style['color'],
                           alpha=0.18, s=14, zorder=1, linewidth=0)
        ax.errorbar(x, y, yerr=e, label=treat, capsize=3, lw=2, markersize=6, zorder=3, **style)
        group_means[treat] = y

    if stats_table is not None:
        stab = stats_table[(stats_table.domain == feature) & (stats_table.sex == sex)]
        for xi, c in enumerate(combos):
            srow = stab[_combo_mask(stab, cols, c)]
            if len(srow) and srow['p'].iloc[0] < 0.05:
                ytop = np.nanmax([group_means['saline'][xi], group_means['MDMA'][xi]])
                ax.annotate('*', (xi, ytop), xytext=(0, 4), textcoords='offset points',
                            ha='center', fontsize=13, color=COLOR_INK, fontweight='bold')

    for xb in _day_boundaries(combos, cols):
        ax.axvline(xb, color=COLOR_MUTED, lw=0.6, alpha=0.35, zorder=0)

    ax.axhline(0, color=COLOR_MUTED, lw=0.5, ls=':')
    ax.set_xticks(x)
    ax.set_xticklabels([_combo_label(c, cols, active_start_hour) for c in combos], fontsize=7.5)
    ax.set_title(_domain_label(feature), fontsize=10.5)
    sns.despine(ax=ax)
    return ax


def plot_pooled_treatment_progression_grid(window_deltas, features, sexes=('female', 'male'),
                                            window_col='day', stats_table=None, title=None,
                                            save_path=None, show=True, dpi=150):
    """Reference grid: one row per sex, one column per feature, each panel via
    plot_pooled_treatment_progression(). '*' marks windows where compute_pooled_progression()
    found nominal p<.05 (MDMA vs saline, background pooled). Pass
    window_col=['day', 'time_window'] (with a 'day'-resolved `window_deltas`) for the
    day-by-day within-session view.

    Returns: (fig, axes)
    """
    cols = [window_col] if isinstance(window_col, str) else list(window_col)
    n_x = len(_ordered_combos(window_deltas.dropna(subset=[f'{features[0]}__delta']), cols))
    panel_w = 2.8 if n_x <= 3 else 3.5
    fig, axes = plt.subplots(len(sexes), len(features),
                              figsize=(panel_w * len(features), 3.9 * len(sexes)), squeeze=False)

    for row, sex in enumerate(sexes):
        for col, feat in enumerate(features):
            ax = axes[row, col]
            plot_pooled_treatment_progression(window_deltas, feat, sex, ax=ax,
                                               window_col=window_col, stats_table=stats_table)
            ax.set_title(_domain_label(feat), fontsize=9) if row == 0 else ax.set_title('')
            if col == 0:
                ax.set_ylabel(f'{sex}\nΔ z-score (post − baseline)', fontsize=9)

    axes[0, -1].legend(fontsize=8, loc='upper right')
    if title:
        fig.suptitle(title, color=COLOR_INK, fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes


# ============================================================================
# 1. BACKGROUND x TREATMENT INTERACTION
# ============================================================================
FORMULA_RHS = "C(background, Treatment('CTRL')) * C(treatment, Treatment('saline'))"
TERM_LABELS = {
    "C(background, Treatment('CTRL'))[T.ELA]": 'background_ELA',
    "C(treatment, Treatment('saline'))[T.MDMA]": 'treatment_MDMA',
    "C(background, Treatment('CTRL'))[T.ELA]:C(treatment, Treatment('saline'))[T.MDMA]": 'background_x_treatment',
}


def run_domain_mixedmodels(deltas, features, sexes=('female', 'male'), group_col='box_ID'):
    """Fit `delta ~ background * treatment` (random intercept = box) once per domain per sex.
    Input: output of aggregate.mouse_level_deltas() (needs '<feature>__delta' columns plus
    'background', 'treatment', 'sex', group_col).

    Returns a tidy long table: domain, sex, term, coef, se, p, n, p_fdr - one row per
    (domain, sex, term) with term in {background_ELA, treatment_MDMA, background_x_treatment}.
    p_fdr corrects across domains WITHIN each (sex, term) group (i.e. the "N domains x 2 sexes"
    multiple-testing scope called out in the preliminary analysis, not across terms too).
    """
    rows = []
    for sex in sexes:
        sub = deltas[deltas.sex == sex]
        for feat in features:
            y_col = f'{feat}__delta'
            if y_col not in sub.columns:
                continue
            try:
                fit = run_mixedlm(sub, y_col=y_col, formula_rhs=FORMULA_RHS, group_col=group_col)
            except Exception as e:
                print(f"[run_domain_mixedmodels] {feat} ({sex}): fit failed ({e}), skipped")
                continue
            n = int(fit.nobs)
            for raw_term, label in TERM_LABELS.items():
                if raw_term not in fit.params.index:
                    continue
                rows.append(dict(domain=feat, sex=sex, term=label,
                                  coef=fit.params[raw_term], se=fit.bse[raw_term],
                                  p=fit.pvalues[raw_term], n=n))
    mm = pd.DataFrame(rows)
    if len(mm):
        mm['p_fdr'] = np.nan
        for (sex, term), idx in mm.groupby(['sex', 'term']).groups.items():
            mm.loc[idx, 'p_fdr'] = multipletests(mm.loc[idx, 'p'], method='fdr_bh')[1]
    return mm


def plot_bg_x_treatment_interaction(deltas, mm, sex, features=None, n_panels=4,
                                     show_individual=True, save_path=None, show=True, dpi=150):
    """Interaction plot(s): x = treatment (saline, MDMA), one line per background (CTRL/ELA),
    y = mean +/- SEM delta (post-injection - baseline). One panel per domain.

    This is the generalized, reusable version of the old fig5_interaction.py 4-panel figure:
    same visual language, but panels/titles are picked from live data (mm, from
    run_domain_mixedmodels()) instead of hardcoded feature names and manually copy-pasted
    p-values that go stale the moment the domain set or data changes.

    Args:
        deltas: output of aggregate.mouse_level_deltas() (as passed to run_domain_mixedmodels).
        mm: output of run_domain_mixedmodels() - supplies the p-value annotations.
        sex: 'female' or 'male'.
        features: explicit list of domains to plot; if None, auto-picks the `n_panels` domains
            with the lowest background_x_treatment p-value for this sex.
        show_individual: overlay faint jittered per-mouse points behind the mean+/-SEM lines,
            so individual-animal variability in the drug response is visible, not just the mean.
        save_path: if given, saves the figure there (parent dir created if needed).

    Returns: (fig, axes)
    """
    if features is None:
        cand = mm[(mm.sex == sex) & (mm.term == 'background_x_treatment')].sort_values('p')
        features = cand['domain'].head(n_panels).tolist()

    order = ['saline', 'MDMA']
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(3.7 * n, 4.5), squeeze=False)
    axes = axes[0]

    rng = np.random.default_rng(0)
    for ax, feat in zip(axes, features):
        col = f'{feat}__delta'
        sub = deltas[deltas.sex == sex][['background', 'treatment', col]].dropna()
        means = sub.groupby(['background', 'treatment'])[col].mean().unstack()
        sems = sub.groupby(['background', 'treatment'])[col].sem().unstack()

        for bg, style, xoff in [('CTRL', dict(color=COLOR_CTRL, marker='o'), -0.12),
                                 ('ELA', dict(color=SEX_ELA_COLORS.get(sex, '#e34948'), marker='s'), 0.12)]:
            if bg not in means.index:
                continue
            if show_individual:
                for ti, t in enumerate(order):
                    pts = sub.loc[(sub.background == bg) & (sub.treatment == t), col]
                    if len(pts):
                        jitter = rng.uniform(-0.06, 0.06, size=len(pts))
                        ax.scatter(np.full(len(pts), ti) + xoff + jitter, pts,
                                   color=style['color'], alpha=0.25, s=16, zorder=1, linewidth=0)
            y = means.loc[bg, order].values
            e = sems.loc[bg, order].values
            ax.errorbar(np.arange(2), y, yerr=e, label=bg, capsize=4, lw=2, markersize=8,
                        zorder=3, **style)

        ax.set_xticks(np.arange(2))
        ax.set_xticklabels(order)
        ax.axhline(0, color=COLOR_MUTED, lw=0.5, ls=':')

        title = _domain_label(feat)
        rows = mm[(mm.domain == feat) & (mm.sex == sex)]
        p_txt = []
        for term, short in [('treatment_MDMA', 'MDMA'), ('background_ELA', 'ELA'),
                             ('background_x_treatment', 'ELA×MDMA')]:
            r = rows[rows.term == term]
            if len(r):
                p_txt.append(f"{short} p={r['p'].iloc[0]:.3f}")
        if p_txt:
            title += '\n' + ', '.join(p_txt)
        ax.set_title(title, fontsize=9.5)

        if ax is axes[0]:
            ax.set_ylabel('Δ (post-injection − baseline)\nz-scored composite')
            ax.legend(fontsize=9)
        sns.despine(ax=ax)

    fig.suptitle(f'Does ELA modulate the behavioral response to MDMA? ({sex})\n'
                 'mixed model: Δ ~ background × treatment, random=box',
                 fontsize=12, fontweight='bold', y=1.1)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes


TERM_DISPLAY = {
    'treatment_MDMA': 'MDMA main effect (treatment)',
    'background_ELA': 'ELA main effect (background)',
    'background_x_treatment': 'ELA × MDMA interaction',
}


def plot_mixedmodel_null_check(mm, term='background_x_treatment', term_label=None,
                                sexes=('female', 'male'), sig_thresh=0.05, title=None,
                                save_path=None, show=True, dpi=150):
    """Simplest possible summary of one mixed-model term: every domain x sex nominal p-value for
    that term plotted as a single dot on ONE shared 0-1 axis (one row per sex), against the
    conventional p=.05 line - deliberately NOT another forest/dumbbell/volcano chart, since the
    point of this view is one clear statement ("is there a clear effect at all?"), not a
    domain-by-domain breakdown (see plot_mixedmodel_forest() for that level of detail). Works
    equally for the MDMA main effect ('treatment_MDMA') or the ELA x MDMA interaction
    ('background_x_treatment') - call it once per term to present them side by side.

    The headline isn't the p-values individually, it's how many cross the line vs how many
    WOULD cross it by chance alone at this many independent tests (n_domains x n_sexes x
    sig_thresh) - annotated directly on the plot, since a "no clear effect" claim is about that
    comparison, not about any single domain.

    Args:
        mm: output of run_domain_mixedmodels() (whole-session; filtered to `term` internally).
        term: which mixed-model term to summarize - see TERM_DISPLAY for the recognized set.
        term_label: axis label override; defaults to TERM_DISPLAY.get(term, term).

    Returns: (fig, ax)
    """
    sub = mm[mm.term == term].copy()
    n_tests = len(sub)
    n_hit = int((sub['p'] < sig_thresh).sum())
    n_fdr = int((sub['p_fdr'] < sig_thresh).sum())
    expected = n_tests * sig_thresh
    label = term_label or TERM_DISPLAY.get(term, term)

    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    rng = np.random.default_rng(0)
    y_pos = {sex: len(sexes) - 1 - i for i, sex in enumerate(sexes)}

    for sex in sexes:
        s = sub[sub.sex == sex]
        color = SEX_ELA_COLORS.get(sex, COLOR_MUTED)
        y0 = y_pos.get(sex, 0)
        y = y0 + rng.uniform(-0.14, 0.14, size=len(s))
        hit = (s['p'] < sig_thresh).values
        ax.scatter(s.loc[~hit, 'p'], y[~hit], facecolor='white', edgecolor=color,
                   s=75, linewidth=1.4, zorder=3)
        ax.scatter(s.loc[hit, 'p'], y[hit], color=color, s=90, zorder=4,
                   edgecolor='white', linewidth=0.8)

    ax.axvline(sig_thresh, color=COLOR_MUTED, lw=1.2, ls='--', zorder=1)
    ax.text(sig_thresh, len(sexes) - 0.42, f'p = {sig_thresh}', fontsize=9.5,
            color=COLOR_MUTED, ha='center')
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.6, len(sexes) - 0.5)
    ax.set_yticks([y_pos[s] for s in sexes])
    ax.set_yticklabels(list(sexes), fontsize=12)
    ax.set_xlabel(f'nominal p-value ({label}, all 12 core domains)', fontsize=11)
    sns.despine(ax=ax, left=True)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color=COLOR_MUTED, alpha=0.2, linewidth=0.5)

    note = (f'{n_hit} of {n_tests} domain × sex tests reach nominal p < {sig_thresh}  '
            f'(≈{expected:.1f} expected by chance alone at this many tests)  ·  '
            f'{n_fdr} survive FDR correction')
    fig.text(0.5, 0.045, note, ha='center', va='bottom', fontsize=11, color=COLOR_INK,
              fontweight='bold')

    if title:
        fig.suptitle(title, color=COLOR_INK, fontsize=14)
    plt.tight_layout(rect=(0, 0.16, 1, 1))

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    return fig, ax


def plot_mixedmodel_forest(mm, term='background_x_treatment', sexes=('female', 'male'),
                            feature_order=None, sig_thresh=0.05, xlim=None, title=None,
                            save_path=None, show=True, dpi=150):
    """All-domain overview forest plot of one mixed-model term - domain-labeled, effect size AND
    significance both visible at once (unlike plot_mixedmodel_null_check()'s plain p-value
    strip, which shows significance only). One panel per sex.

    THREE-tier significance (not two): filled = survives FDR correction; ring (colored edge,
    hollow) = nominal p<sig_thresh only; faint (grey edge, hollow) = not even nominal - so a
    "some effect, but not a broad/robust one" pattern (a couple of solid dots among many faint
    ones) reads directly off the chart, without needing a separate summary stat.

    Args:
        mm: output of run_domain_mixedmodels().
        term: one of 'background_ELA', 'treatment_MDMA', 'background_x_treatment'.
        feature_order: top-to-bottom domain order shared across panels; defaults to the order
            domains first appear in mm.
        xlim: shared (lo, hi) x-range override - pass the SAME xlim when calling this once per
            term (e.g. main effect vs interaction) so the two resulting charts are on a fair,
            directly comparable scale instead of each auto-scaling to its own data.

    Returns: (fig, axes)
    """
    sub_all = mm[mm.term == term]
    order = feature_order if feature_order is not None else list(dict.fromkeys(sub_all['domain']))

    fig, axes = plt.subplots(1, len(sexes),
                              figsize=(6.5 * len(sexes), max(4, 0.45 * len(order))),
                              sharex=True, squeeze=False)
    axes = axes[0]

    for ax, sex in zip(axes, sexes):
        res = (sub_all[sub_all.sex == sex].set_index('domain').reindex(order)
               .dropna(subset=['coef']).reset_index())
        y = np.arange(len(res))
        colors = np.where(res['coef'] >= 0, '#e34948', '#2a78d6')
        fdr_sig = (res['p_fdr'] < sig_thresh).values
        nom_sig = (res['p'] < sig_thresh).values & ~fdr_sig
        ns = ~fdr_sig & ~nom_sig

        ax.hlines(y, 0, res['coef'], color=colors, linewidth=1.2, alpha=0.55, zorder=1)
        ax.scatter(res.loc[fdr_sig, 'coef'], y[fdr_sig], color=colors[fdr_sig], s=95, zorder=3,
                   edgecolor=COLOR_INK, linewidth=0.8)
        ax.scatter(res.loc[nom_sig, 'coef'], y[nom_sig], facecolor='white', edgecolor=colors[nom_sig],
                   linewidth=1.8, s=75, zorder=2)
        ax.scatter(res.loc[ns, 'coef'], y[ns], facecolor='white', edgecolor=COLOR_MUTED,
                   linewidth=1.1, s=55, zorder=2, alpha=0.8)
        ax.axvline(0, color=COLOR_MUTED, linewidth=1, linestyle='--', zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([_domain_label(f) for f in res['domain']], fontsize=11)
        ax.invert_yaxis()
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_xlabel(f'{TERM_DISPLAY.get(term, term)} coefficient', fontsize=11)
        n_txt = res["n"].iloc[0] if len(res) else "-"
        n_fdr, n_nom = int(fdr_sig.sum()), int(nom_sig.sum())
        ax.set_title(f'{sex} (n={n_txt} mice) — {n_fdr} FDR-sig, {n_nom} nominal-only', fontsize=12.5)
        ax.tick_params(axis='y', length=0)
        sns.despine(ax=ax, left=True)
        ax.grid(axis='x', color=COLOR_MUTED, alpha=0.25, linewidth=0.5)

    handles = [
        plt.Line2D([0], [0], marker='o', markerfacecolor=COLOR_MUTED, markeredgecolor=COLOR_INK,
                   linestyle='', markersize=10, label=f'FDR p < {sig_thresh}'),
        plt.Line2D([0], [0], marker='o', markerfacecolor='white', markeredgecolor=COLOR_MUTED,
                   markeredgewidth=1.8, linestyle='', markersize=10, label=f'nominal p < {sig_thresh}'),
        plt.Line2D([0], [0], marker='o', markerfacecolor='white', markeredgecolor=COLOR_MUTED,
                   linestyle='', markersize=8, alpha=0.8, label='ns'),
        plt.Line2D([0], [0], marker='s', color='#e34948', linestyle='', markersize=9,
                   label='coef ≥ 0'),
        plt.Line2D([0], [0], marker='s', color='#2a78d6', linestyle='', markersize=9,
                   label='coef < 0'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.05))
    if title:
        fig.suptitle(title, color=COLOR_INK, fontsize=14)
    plt.tight_layout(rect=(0, 0.06, 1, 1))

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes


# ============================================================================
# 2. NORMALIZATION - does MDMA move ELA animals TOWARD the CTRL level?
# ============================================================================
def compute_normalization_table(mouse_sess, features, sexes=('female', 'male')):
    """For each domain x sex x treatment-group, box-paired ELA-vs-CTRL effect size (Cohen's dz)
    at baseline and again post-injection, then how much that gap shrank.

    gap_shrink = |dz_baseline| - |dz_post|  (positive = ELA-CTRL gap narrowed after injection)
    normalization_index = gap_shrink in MDMA-dosed boxes - gap_shrink in saline-dosed boxes
        (positive = the gap narrowed specifically because of MDMA, beyond whatever narrowing
        saline-dosed boxes show on retest alone - the therapeutic-direction signature).

    Input: mouse_sess - output of aggregate_to_mouse + compute_domain_scores(reference_mask=
    baseline) with 'time_point' (baseline/MDMA session) and 'treatment' (MDMA/saline, box-level)
    both present, domain scores on a shared baseline-referenced scale (as built in run_analysis.py
    step 7 before mouse_level_deltas()).

    Returns one row per (feature, sex): dz_baseline_MDMAbox, dz_post_MDMAbox,
    dz_baseline_salinebox, dz_post_salinebox, shrink_MDMA, shrink_saline, normalization_index.
    """
    rows = []
    for sex in sexes:
        for treat_group in ['MDMA', 'saline']:
            for time_point, tag in [('baseline', 'baseline'), ('MDMA', 'post')]:
                grp_sub = mouse_sess[(mouse_sess.sex == sex) & (mouse_sess.treatment == treat_group)
                                      & (mouse_sess.time_point == time_point)]
                wide = box_paired_wide(grp_sub, features=features, extra_group_cols=())
                res = paired_effect_table(wide, features)
                if len(res):
                    res = res[['feature', 'cohen_dz', 'n_pairs']].copy()
                    res['sex'] = sex
                    res['treatment_group'] = treat_group
                    res['session'] = tag
                    rows.append(res)
    long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not len(long):
        return long

    piv = long.pivot_table(index=['feature', 'sex', 'treatment_group'],
                            columns='session', values=['cohen_dz', 'n_pairs'])
    piv.columns = [f'{val}_{sess}' for val, sess in piv.columns]
    piv = piv.reset_index()
    piv['gap_shrink'] = piv['cohen_dz_baseline'].abs() - piv['cohen_dz_post'].abs()

    mdma = piv[piv.treatment_group == 'MDMA'].set_index(['feature', 'sex'])
    saline = piv[piv.treatment_group == 'saline'].set_index(['feature', 'sex'])
    idx = mdma.index.intersection(saline.index)

    norm = pd.DataFrame({
        'dz_baseline_MDMAbox': mdma.loc[idx, 'cohen_dz_baseline'],
        'dz_post_MDMAbox': mdma.loc[idx, 'cohen_dz_post'],
        'dz_baseline_salinebox': saline.loc[idx, 'cohen_dz_baseline'],
        'dz_post_salinebox': saline.loc[idx, 'cohen_dz_post'],
        'shrink_MDMA': mdma.loc[idx, 'gap_shrink'],
        'shrink_saline': saline.loc[idx, 'gap_shrink'],
    }).reset_index()
    norm['normalization_index'] = norm['shrink_MDMA'] - norm['shrink_saline']
    return norm.sort_values('normalization_index', ascending=False).reset_index(drop=True)


def plot_normalization(norm_table, sex, feature_order=None, save_path=None, show=True, dpi=150,
                        title=None):
    """Before -> after dot-and-line plot: for each domain, where the ELA-CTRL gap (Cohen's dz)
    sits at baseline vs. post-injection, separately for MDMA-dosed boxes (solid) and
    saline-dosed boxes (dashed, retest-effect reference). A domain shows a normalization
    (therapeutic-direction) pattern when the MDMA-box segment moves toward 0 by MORE than the
    saline-box segment does.

    Returns: (fig, ax)
    """
    sub = norm_table[norm_table.sex == sex]
    if feature_order is None:
        feature_order = list(dict.fromkeys(
            norm_table.sort_values('normalization_index', ascending=False)['feature']))
    sub = sub.set_index('feature').reindex(feature_order).dropna(how='all').reset_index()

    y = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(7.5, max(4, 0.5 * len(sub))))

    for off, grp, base_col, post_col, color, ls in [
        (-0.15, 'MDMA', 'dz_baseline_MDMAbox', 'dz_post_MDMAbox', COLOR_MDMA_GROUP, '-'),
        (0.15, 'saline', 'dz_baseline_salinebox', 'dz_post_salinebox', COLOR_SALINE_GROUP, '--'),
    ]:
        for i, row in sub.iterrows():
            b, p = row[base_col], row[post_col]
            if pd.isna(b) or pd.isna(p):
                continue
            yy = y[i] + off
            ax.plot([b, p], [yy, yy], color=color, lw=2, ls=ls, zorder=2)
            ax.scatter(b, yy, facecolor='white', edgecolor=color, s=45, zorder=3, linewidth=1.5)
            marker = '>' if p >= b else '<'
            ax.scatter(p, yy, facecolor=color, edgecolor=color, s=55, zorder=3, marker=marker)

    ax.axvline(0, color=COLOR_MUTED, lw=1, ls=':', zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([_domain_label(f) for f in sub['feature']], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Cohen's dz (ELA − CTRL, box-paired)\n"
                  "hollow = baseline session, arrow = post-injection session", fontsize=10)
    ax.set_title(title or f'Does MDMA narrow the ELA-CTRL gap? ({sex})', fontsize=13)
    handles = [
        plt.Line2D([0], [0], color=COLOR_MDMA_GROUP, lw=2, label='MDMA-dosed boxes'),
        plt.Line2D([0], [0], color=COLOR_SALINE_GROUP, lw=2, ls='--', label='saline-dosed boxes (retest ref.)'),
    ]
    ax.legend(handles=handles, loc='best', fontsize=9)
    sns.despine(ax=ax, left=True)
    ax.grid(axis='x', color=COLOR_MUTED, alpha=0.25, linewidth=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    return fig, ax


def compute_normalization_gaps(mouse_sess, features, sexes=('female', 'male')):
    """Per-box ELA-CTRL gap (raw box-paired difference on the z-scored composite scale, NOT the
    standardized Cohen's dz that compute_normalization_table() summarizes to one number per
    group) at baseline and post-injection, in both MDMA-dosed and saline-dosed boxes - the
    per-box distribution underlying that table, so the actual box-to-box spread is visible
    rather than just a group-level effect size. Feeds plot_normalization_gap_grid()'s
    one-panel-per-domain view.

    Input: mouse_sess - same as compute_normalization_table() (needs 'time_point', 'treatment',
    'sex', 'box_ID', 'background' plus `features` on a shared baseline-referenced scale).

    Returns long tidy table: feature, sex, treatment_group ('MDMA'/'saline'), session
    ('baseline'/'post'), box_ID, gap (= ELA box mean - CTRL box mean).
    """
    rows = []
    for sex in sexes:
        for treat_group in ['MDMA', 'saline']:
            for time_point, tag in [('baseline', 'baseline'), ('MDMA', 'post')]:
                grp_sub = mouse_sess[(mouse_sess.sex == sex) & (mouse_sess.treatment == treat_group)
                                      & (mouse_sess.time_point == time_point)]
                wide = box_paired_wide(grp_sub, features=features, extra_group_cols=())
                for feat in features:
                    ela_col, ctrl_col = f'{feat}_ELA', f'{feat}_CTRL'
                    if ela_col not in wide.columns or ctrl_col not in wide.columns:
                        continue
                    sub = wide[['box_ID', ela_col, ctrl_col]].dropna()
                    if not len(sub):
                        continue
                    rows.append(pd.DataFrame({
                        'feature': feat, 'sex': sex, 'treatment_group': treat_group,
                        'session': tag, 'box_ID': sub['box_ID'],
                        'gap': sub[ela_col] - sub[ctrl_col],
                    }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=['feature', 'sex', 'treatment_group', 'session', 'box_ID', 'gap'])


def plot_normalization_gap(gaps, feature, sex, ax=None, show_legend=True):
    """One domain's box-paired ELA-CTRL gap, x-axis = treatment_group (saline, MDMA - i.e. which
    substance the box received), each split (dodged) into baseline vs post-injection so the
    within-box-group shift is visible directly, without collapsing to a single effect-size
    number first (companion, per-box-distribution view to plot_normalization()).

    Input: gaps - output of compute_normalization_gaps().
    Returns: ax
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3.6, 4.3))

    sub = gaps[(gaps.feature == feature) & (gaps.sex == sex)]
    x_order = ['saline', 'MDMA']
    session_order = ['baseline', 'post']
    palette = {'baseline': COLOR_MUTED, 'post': SEX_ELA_COLORS.get(sex, '#e34948')}

    sns.boxplot(data=sub, x='treatment_group', y='gap', hue='session', order=x_order,
                hue_order=session_order, palette=palette, showfliers=False, width=0.6, ax=ax)
    for patch in ax.patches:
        patch.set_alpha(0.45)
    sns.stripplot(data=sub, x='treatment_group', y='gap', hue='session', order=x_order,
                  hue_order=session_order, palette=palette, dodge=True, size=4.5, jitter=0.12,
                  linewidth=0.3, edgecolor='white', legend=False, ax=ax)
    for coll in ax.collections:
        coll.set_alpha(0.85)

    ax.axhline(0, color=COLOR_MUTED, lw=0.6, ls=':', zorder=0)
    ax.set_xlabel('')
    ax.set_title(_domain_label(feature), fontsize=10.5)
    ax.set_ylabel('')
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            seen.setdefault(l, h)
        ax.legend(seen.values(), seen.keys(), fontsize=7.5, loc='best', title=None)
    else:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    sns.despine(ax=ax)
    return ax


def plot_normalization_gap_grid(gaps, features, sexes=('female', 'male'), pair_sexes=False,
                                 n_cols=None, legend_loc='top', title=None, save_path=None,
                                 show=True, dpi=150):
    """Grid of plot_normalization_gap() panels - the "does MDMA narrow the ELA-CTRL gap"
    question at the box level rather than as one dz number per group (see plot_normalization()
    for that summary view).

    Two layouts:
    - pair_sexes=False (default): one row per sex, one column per domain.
    - pair_sexes=True (requires exactly two sexes): each domain gets its male/female panels
        placed side by side instead, with a shared y-axis range per domain pair (the union of
        their individual auto-scaled ranges) so the two sexes are directly comparable at a
        glance, and the right panel's y tick labels suppressed since it shares the left
        panel's scale - same convention as e.g. temporal_dynamics.plot_temporal_grid's
        pair_sexes option.

    n_cols: how many domains (pair_sexes=True) or feature-columns (pair_sexes=False) to place
        per row before wrapping into an additional row-group, instead of always laying every
        domain out in one very wide row (the default, n_cols=None -> len(features), i.e. no
        wrapping - the original single-row layout). Useful once `features` is long (e.g. all
        12 CORE_DOMAIN_FEATURES).

    legend_loc: 'top' (default) / 'bottom' draws ONE figure-level legend above/below the whole
        grid, instead of inside any single subplot's corner - it always shows one shared
        'baseline' entry plus one 'post (sex)' entry per sex (the post color is sex-specific,
        SEX_ELA_COLORS, so a single-sex per-panel legend would misrepresent the other sex's
        color). Pass None to omit it.

    Input: gaps - output of compute_normalization_gaps().
    Returns: (fig, axes)
    """
    ylabel = 'Δ (ELA − CTRL, box-paired)\nz-scored composite'
    n_feat = len(features)
    ncols = min(n_feat, n_cols or n_feat)
    n_groups = int(np.ceil(n_feat / ncols))

    if pair_sexes:
        if len(sexes) != 2:
            raise ValueError('pair_sexes=True requires exactly two sexes')
        sex_left, sex_right = sexes
        nrows = n_groups
        fig, axes = plt.subplots(nrows, ncols * 2,
                                  figsize=(2.6 * ncols * 2, 4.4 * nrows), squeeze=False)

        for i, feat in enumerate(features):
            row, col = divmod(i, ncols)
            ax_left, ax_right = axes[row, col * 2], axes[row, col * 2 + 1]
            plot_normalization_gap(gaps, feat, sex_left, ax=ax_left, show_legend=False)
            plot_normalization_gap(gaps, feat, sex_right, ax=ax_right, show_legend=False)

            lo = min(ax_left.get_ylim()[0], ax_right.get_ylim()[0])
            hi = max(ax_left.get_ylim()[1], ax_right.get_ylim()[1])
            ax_left.set_ylim(lo, hi)
            ax_right.set_ylim(lo, hi)
            ax_right.tick_params(labelleft=False)

            domain_label = _domain_label(feat)
            ax_left.set_title(f'{domain_label}\n{sex_left}', fontsize=8.5)
            ax_right.set_title(f'{domain_label}\n{sex_right}', fontsize=8.5)
            if col == 0:
                ax_left.set_ylabel(ylabel, fontsize=9)

        for i in range(n_feat, nrows * ncols):
            row, col = divmod(i, ncols)
            axes[row, col * 2].axis('off')
            axes[row, col * 2 + 1].axis('off')

    else:
        nrows = n_groups * len(sexes)
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 4.0 * nrows), squeeze=False)

        for grp_i in range(n_groups):
            feats_in_group = features[grp_i * ncols:(grp_i + 1) * ncols]
            for row_offset, sex in enumerate(sexes):
                r = grp_i * len(sexes) + row_offset
                for col, feat in enumerate(feats_in_group):
                    ax = axes[r, col]
                    plot_normalization_gap(gaps, feat, sex, ax=ax, show_legend=False)
                    ax.set_title(_domain_label(feat) if row_offset == 0 else '', fontsize=9.5)
                    ax.set_ylabel(f'{sex}\n{ylabel}' if col == 0 else '', fontsize=9)
                for col in range(len(feats_in_group), ncols):
                    axes[r, col].axis('off')

    top_pad, bottom_pad, title_y, legend_y = _legend_layout(legend_loc, title)
    plt.tight_layout(rect=(0, bottom_pad, 1, 1 - top_pad))
    fig.suptitle(title or 'Does MDMA narrow the ELA-CTRL gap? (box-paired, per domain)',
                 color=COLOR_INK, fontsize=13, fontweight='bold', y=title_y)

    if legend_loc and legend_y is not None:
        handles = [Patch(facecolor=COLOR_MUTED, alpha=0.45, label='baseline')]
        handles += [Patch(facecolor=SEX_ELA_COLORS.get(sex, '#e34948'), alpha=0.45,
                           label=f'post ({sex})') for sex in sexes]
        fig.legend(handles=handles, loc='center', bbox_to_anchor=(0.5, legend_y),
                   ncol=len(handles), frameon=False, fontsize=9)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes


# ============================================================================
# 3. TIME-COURSE - small (fine-resolution bin) vs large (whole session) windows
# ============================================================================
def compute_window_deltas(df_fine, domain_defs, id_cols, phase='active',
                           session_col='time_point', session_values=('baseline', 'MDMA'),
                           window_col='time_window'):
    """Per-mouse-per-window delta (post-injection - baseline), z-scored WITHIN each time window
    separately (reference = that window's own baseline distribution) so a within-session
    circadian/fatigue trend in the raw features doesn't get mistaken for a treatment effect.

    Input: df_fine - a df loaded from a finer-than-12h QC_table file (1h/2h/3h/4h/6h), with
    both time_point values and phase=='active' rows present (e.g. the `df_3h` already loaded in
    run_analysis.py step 6).

    Returns: output of aggregate.mouse_level_deltas(), with `window_col` retained as an extra
    index column alongside the usual id_cols.
    """
    sub = df_fine[df_fine['phase'] == phase].copy()
    raw_features = sorted({f for spec in domain_defs.values()
                            for f in spec['features'] + spec.get('flip', [])})
    # normDS/rank (used by social_hierarchy_score) don't exist as columns below 12h resolution
    # (not NaN - genuinely absent), so they must be dropped from the aggregation call itself;
    # compute_domain_scores() below already tolerates missing features per-domain.
    missing = [f for f in raw_features if f not in sub.columns]
    if missing:
        print(f"[compute_window_deltas] {missing} not available at this resolution, skipping "
              f"those features (domains using them are scored on their remaining features only)")
    raw_features = [f for f in raw_features if f in sub.columns]

    mouse_win = aggregate_to_mouse(sub, id_cols + [session_col, window_col], raw_features)
    mouse_win['__zgrp'] = mouse_win['sex'].astype(str) + '__' + mouse_win[window_col].astype(str)
    baseline_mask = mouse_win[session_col] == session_values[0]
    mouse_win = compute_domain_scores(mouse_win, domain_defs=domain_defs,
                                       reference_mask=baseline_mask, group_col='__zgrp')

    return mouse_level_deltas(mouse_win, features=list(domain_defs.keys()),
                               id_cols=id_cols + [window_col], session_col=session_col,
                               session_values=session_values)


def _window_sort_key(w):
    return int(w.split('-')[0])


def _window_label(w, active_start_hour=12):
    lo, hi = (int(x) for x in w.split('-'))
    return f'{lo - active_start_hour}-{hi - active_start_hour}h'


def _ordered_combos(df, cols):
    """Sorted unique combinations of `cols` present in df, as a list of tuples (column order
    preserved). 'day' sorts as int, 'time_window' sorts by its numeric start hour
    (_window_sort_key); any other column sorts lexically. Shared x-axis builder for
    plot_delta_by_window() and plot_pooled_treatment_progression() - both accept `window_col` as
    a single column name (e.g. 'time_window') OR a list (e.g. ['day', 'time_window']) so a
    within-session window can be resolved PER treatment day instead of silently averaging the
    same clock-hour bin together across all 3 post-treatment days (which hides day-to-day
    change, e.g. tolerance/sensitization, inside what looks like a single time-course)."""
    combos = df[cols].drop_duplicates()

    def key(row):
        out = []
        for c, v in zip(cols, row):
            if c == 'day':
                out.append(int(v))
            elif c == 'time_window':
                out.append(_window_sort_key(v))
            else:
                out.append(v)
        return tuple(out)

    return sorted((tuple(r) for r in combos.itertuples(index=False)), key=key)


def _combo_label(combo, cols, active_start_hour=12):
    parts = []
    for c, v in zip(cols, combo):
        if c == 'day':
            parts.append(f'Day {v}')
        elif c == 'time_window':
            parts.append(_window_label(v, active_start_hour))
        else:
            parts.append(str(v))
    return '\n'.join(parts)


def _combo_mask(df, cols, combo):
    m = pd.Series(True, index=df.index)
    for c, v in zip(cols, combo):
        m &= (df[c] == v)
    return m


def _day_boundaries(combos, cols):
    """x-positions (i - 0.5, between ticks) where `day` changes - a light vertical separator,
    only meaningful when 'day' is combined with a second column (e.g. 'time_window')."""
    if 'day' not in cols or len(cols) < 2:
        return []
    di = cols.index('day')
    return [i - 0.5 for i in range(1, len(combos)) if combos[i][di] != combos[i - 1][di]]


def plot_delta_by_window(window_deltas, feature, sex, ax=None, window_col='time_window',
                          full_session_deltas=None, active_start_hour=12):
    """One domain's MDMA-vs-baseline delta across fine time windows (solid, mean+/-SEM, one
    line per background), with the same-window saline-box trajectory as a faint dotted
    reference (non-specific session-to-session drift), and - if `full_session_deltas` is given -
    a dashed horizontal line at the whole-session (large-window) mean delta for comparison, so
    the small-window time-course and the large-window headline number sit in the same panel.

    window_col: a single column name (e.g. the default 'time_window', which pools all
    post-treatment days together within each clock-hour bin - only reasonable if `window_deltas`
    itself has no 'day' column, i.e. was built without 'day' in compute_window_deltas()'s
    id_cols), or a list e.g. ['day', 'time_window'] to instead show each treatment day's
    within-session time-course separately (requires `window_deltas` built WITH 'day' in
    compute_window_deltas()'s id_cols) - light vertical rules mark day boundaries.

    Returns: ax
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3.6, 4.3))

    cols = [window_col] if isinstance(window_col, str) else list(window_col)
    col = f'{feature}__delta'
    ela_color = SEX_ELA_COLORS.get(sex, '#e34948')

    sub_mdma = window_deltas[(window_deltas.sex == sex) & (window_deltas.treatment == 'MDMA')][
        ['background', *cols, col]].dropna()
    combos = _ordered_combos(sub_mdma, cols)
    x = np.arange(len(combos))

    for bg, style in [('CTRL', dict(color=COLOR_CTRL, marker='o')),
                       ('ELA', dict(color=ela_color, marker='s'))]:
        g = sub_mdma[sub_mdma.background == bg]
        y, e = [], []
        for c in combos:
            vals = g.loc[_combo_mask(g, cols, c), col]
            y.append(vals.mean())
            e.append(vals.sem())
        ax.errorbar(x, y, yerr=e, label=f'{bg} (MDMA)', capsize=3, lw=2, markersize=6,
                    zorder=3, **style)

    sub_sal = window_deltas[(window_deltas.sex == sex) & (window_deltas.treatment == 'saline')][
        ['background', *cols, col]].dropna()
    for bg, color in [('CTRL', COLOR_CTRL), ('ELA', ela_color)]:
        g = sub_sal[sub_sal.background == bg]
        y = [g.loc[_combo_mask(g, cols, c), col].mean() for c in combos]
        ax.plot(x, y, ls=':', lw=1.3, alpha=0.5, color=color, zorder=1)

    if full_session_deltas is not None:
        sub_full = full_session_deltas[(full_session_deltas.sex == sex)
                                        & (full_session_deltas.treatment == 'MDMA')]
        for bg, color in [('CTRL', COLOR_CTRL), ('ELA', ela_color)]:
            v = sub_full.loc[sub_full.background == bg, col].dropna()
            if len(v):
                ax.axhline(v.mean(), color=color, lw=1, ls='--', alpha=0.5, zorder=0)

    for xb in _day_boundaries(combos, cols):
        ax.axvline(xb, color=COLOR_MUTED, lw=0.6, alpha=0.35, zorder=0)

    ax.axhline(0, color=COLOR_MUTED, lw=0.5, ls=':')
    ax.set_xticks(x)
    ax.set_xticklabels([_combo_label(c, cols, active_start_hour) for c in combos], fontsize=7.5)
    if 'day' not in cols:
        ax.set_xlabel('Hours since active-phase start', fontsize=8.5)
    ax.set_title(_domain_label(feature), fontsize=10.5)
    sns.despine(ax=ax)
    return ax


def plot_delta_by_window_grid(window_deltas, features, sexes=('female', 'male'),
                               window_col='time_window', full_session_deltas=None, title=None,
                               save_path=None, show=True, dpi=150):
    """Reference grid: one row per sex, one column per feature, each panel via
    plot_delta_by_window(). Dotted lines = same-window saline-box reference; dashed horizontal
    lines (when full_session_deltas is given) = whole-session mean, for the small-vs-large
    window comparison. Pass window_col=['day', 'time_window'] (with a 'day'-resolved
    `window_deltas`) to show each treatment day's time-course separately instead of pooling all
    post-treatment days into one representative within-session shape.

    Returns: (fig, axes)
    """
    n_cols_x = len(_ordered_combos(window_deltas.dropna(subset=[f'{features[0]}__delta']),
                                    [window_col] if isinstance(window_col, str) else list(window_col)))
    panel_w = 2.8 if n_cols_x <= 3 else 3.5
    fig, axes = plt.subplots(len(sexes), len(features),
                              figsize=(panel_w * len(features), 3.9 * len(sexes)), squeeze=False)

    for row, sex in enumerate(sexes):
        for col, feat in enumerate(features):
            ax = axes[row, col]
            plot_delta_by_window(window_deltas, feat, sex, ax=ax, window_col=window_col,
                                  full_session_deltas=full_session_deltas)
            ax.set_title(_domain_label(feat), fontsize=9) if row == 0 else ax.set_title('')
            if col == 0:
                ax.set_ylabel(f'{sex}\nΔ z-score (post − baseline)', fontsize=9)

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


# ============================================================================
# 4. TIME-EPOCH (ACUTE / DAY-1-LATE / FOLLOW-UP) INTERACTION & NORMALIZATION - does ELA's
# modulation of the MDMA response, and any MDMA-specific gap narrowing, concentrate shortly
# after the SINGLE dose, persist later that same day, or hold up into the drug-free follow-up
# days (days 2-3)? Sections 1 and 2 above answer both questions on the whole-(12h)-session
# delta, collapsed across all 3 recording days; this section re-runs the SAME two questions
# within THREE epochs instead - a data-prep step (compute_epoch_mouse_sess) feeding the
# EXISTING run_domain_mixedmodels()/compute_normalization_table(), not a new statistical method.
#
# IMPORTANT: treatment (MDMA/saline) is a SINGLE DOSE, given once on Day 1 - Days 2-3 are
# drug-free follow-up recording, not repeat dosing. Any "Days 2-3" effect below is a delayed /
# carryover effect of that one dose, not a cumulative/repeated-dosing effect.
#
# Epoch definition (day-aware - NOT pooled across days the way a plain hour-of-day split would
# be): 'acute' = Day 1, hours 0-4 post-injection (immediate pharmacology); 'day1_late' = Day 1,
# hours 4-12 (still dosing day, later in that same session); 'followup' = Days 2-3, averaged
# together (drug-free follow-up, i.e. delayed/carryover effects of the single dose). The
# baseline sessions get the same day+hour split applied, so each epoch has a fair,
# like-for-like reference.
# ============================================================================
EPOCH_ORDER = ('acute', 'day1_late', 'followup')
EPOCH_LABELS = {'acute': 'Acute (Day 1, 0–4h)', 'day1_late': 'Day 1, 4–12h',
                'followup': 'Days 2–3 (drug-free follow-up)'}
EPOCH_MARKERS = {'acute': '^', 'day1_late': 'D', 'followup': 'o'}
EPOCH_HATCHES = {'acute': '', 'day1_late': '\\\\\\', 'followup': '///'}


def _epoch_from_day_window(day, w, active_start_hour=12, acute_hours=4):
    if day != 1:
        return 'followup'
    lo = int(w.split('-')[0])
    return 'acute' if (lo - active_start_hour) < acute_hours else 'day1_late'


def compute_epoch_mouse_sess(df_fine, domain_defs, id_cols, phase='active',
                              time_points=('baseline', 'MDMA'), active_start_hour=12,
                              acute_hours=4, window_col='time_window', day_col='day'):
    """Mouse-level domain scores tagged by a day-aware 'epoch' - 'acute' (Day 1, the first
    `acute_hours` post-injection), 'day1_late' (Day 1, the rest of that session), or 'followup'
    (Days 2-3, averaged - drug-free follow-up, NOT repeat dosing; treatment is a single dose
    given on Day 1) - see the section-4 module comment above for the full rationale. This
    replaces a plain hour-of-day split (e.g. "0-4h" vs "4-12h" pooled across all 3 days), which
    would silently average the acute window of day 1, 2 and 3 together before any stats run -
    the same day-pooling issue already fixed for compute_window_deltas() in step 5.3.

    Feeds compute_epoch_interaction() and compute_epoch_normalization() below, which each filter
    this table to one epoch and hand it to the EXISTING run_domain_mixedmodels() /
    compute_normalization_table().

    Input: df_fine - a df loaded from a finer-than-12h QC_table file (the same 4h file already
    used for compute_window_deltas() in step 5.3 works directly), with `window_col`, `day_col`
    and both `time_points` present.

    Returns: mouse-level table (one row per mouse per time_point per epoch) with an 'epoch'
    column plus the domain_defs score columns, baseline-referenced per (sex, epoch).
    """
    sub = df_fine[(df_fine['phase'] == phase) & (df_fine['time_point'].isin(time_points))].copy()
    sub['epoch'] = [_epoch_from_day_window(d, w, active_start_hour, acute_hours)
                     for d, w in zip(sub[day_col], sub[window_col])]

    raw_features = sorted({f for spec in domain_defs.values()
                            for f in spec['features'] + spec.get('flip', [])})
    missing = [f for f in raw_features if f not in sub.columns]
    if missing:
        print(f"[compute_epoch_mouse_sess] {missing} not available at this resolution, skipping "
              f"those features (domains using them are scored on their remaining features only)")
    raw_features = [f for f in raw_features if f in sub.columns]

    mouse_ep = aggregate_to_mouse(sub, id_cols + ['time_point', 'epoch'], raw_features)
    mouse_ep['__zgrp'] = mouse_ep['sex'].astype(str) + '__' + mouse_ep['epoch'].astype(str)
    baseline_mask = mouse_ep['time_point'] == time_points[0]
    mouse_ep = compute_domain_scores(mouse_ep, domain_defs=domain_defs,
                                      reference_mask=baseline_mask, group_col='__zgrp')
    return mouse_ep


DAYGROUP_ORDER = ('day1', 'followup')
DAYGROUP_LABELS = {'day1': 'Day 1 (dose)', 'followup': 'Days 2–3 (drug-free follow-up)'}
DAYGROUP_COLOR = {'day1': COLOR_ACUTE, 'followup': COLOR_SUSTAINED}


def compute_daygroup_mouse_sess(df, domain_defs, id_cols, phase='active',
                                 time_points=('baseline', 'MDMA'), day_col='day'):
    """Mouse-level domain scores tagged by 'day1' (the single dosing day) vs 'followup' (days
    2-3, averaged - drug-free follow-up recording, NOT repeat dosing) - the simplified,
    DAY-ONLY version of compute_epoch_mouse_sess() (drops the within-day hour split entirely:
    no 'acute'/'day1_late' distinction, just day1 vs the rest).

    Works directly off the 12h-resolution table (pass the already-loaded `df`, NOT a
    fine-resolution file) - no hour-level binning needed, so every core domain is available on
    its full feature set, including social_hierarchy_score (normDS isn't recorded below 12h
    resolution, so the finer hour-level epoch split has to drop it; this one doesn't).

    Column is still named 'epoch' (values 'day1'/'followup') so this table is a drop-in for
    compute_epoch_interaction() / compute_epoch_normalization() - pass epochs=DAYGROUP_ORDER.

    Returns: mouse-level table (one row per mouse per time_point per day-group) with an 'epoch'
    column plus the domain_defs score columns, baseline-referenced per (sex, day-group).
    """
    sub = df[(df['phase'] == phase) & (df['time_point'].isin(time_points))].copy()
    sub['epoch'] = np.where(sub[day_col] == 1, 'day1', 'followup')

    raw_features = sorted({f for spec in domain_defs.values()
                            for f in spec['features'] + spec.get('flip', [])})
    raw_features = [f for f in raw_features if f in sub.columns]

    mouse_dg = aggregate_to_mouse(sub, id_cols + ['time_point', 'epoch'], raw_features)
    mouse_dg['__zgrp'] = mouse_dg['sex'].astype(str) + '__' + mouse_dg['epoch'].astype(str)
    baseline_mask = mouse_dg['time_point'] == time_points[0]
    mouse_dg = compute_domain_scores(mouse_dg, domain_defs=domain_defs,
                                      reference_mask=baseline_mask, group_col='__zgrp')
    return mouse_dg


def compute_epoch_interaction(mouse_sess_epoch, features, sexes=('female', 'male'),
                               epochs=EPOCH_ORDER, id_cols=None, group_col='box_ID'):
    """Background x treatment interaction (see run_domain_mixedmodels()), fit SEPARATELY within
    each epoch of compute_epoch_mouse_sess() - does ELA modulate the MDMA response differently
    acutely, later the same dosing day, or in the drug-free follow-up days (2-3)?

    Returns tidy long table: domain, sex, epoch, term, coef, se, p, n, p_fdr (p_fdr corrects
    across domains within each sex x term x epoch group - same scoping convention as
    run_domain_mixedmodels(), now also split by epoch).
    """
    if id_cols is None:
        id_cols = ['mouse_ID', 'sex', 'age', 'background', 'treatment', 'box_ID']
    rows = []
    for epoch in epochs:
        sub = mouse_sess_epoch[mouse_sess_epoch.epoch == epoch]
        deltas_ep = mouse_level_deltas(sub, features=features, id_cols=id_cols,
                                        session_col='time_point', session_values=('baseline', 'MDMA'))
        mm = run_domain_mixedmodels(deltas_ep, features, sexes=sexes, group_col=group_col)
        mm['epoch'] = epoch
        rows.append(mm)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compute_epoch_normalization(mouse_sess_epoch, features, sexes=('female', 'male'),
                                 epochs=EPOCH_ORDER):
    """Normalization index (see compute_normalization_table()), computed SEPARATELY within each
    epoch of compute_epoch_mouse_sess() - is any MDMA-specific ELA-CTRL gap narrowing
    concentrated on the single dosing day, or does it hold up / emerge in the drug-free
    follow-up days (2-3)?

    Returns: same columns as compute_normalization_table(), plus an 'epoch' column.
    """
    rows = []
    for epoch in epochs:
        sub = mouse_sess_epoch[mouse_sess_epoch.epoch == epoch]
        norm = compute_normalization_table(sub, features, sexes=sexes)
        norm['epoch'] = epoch
        rows.append(norm)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_epoch_interaction_dumbbell(epoch_mm, feature_order=None, sexes=('female', 'male'),
                                     epochs=EPOCH_ORDER, epoch_labels=None, sig_col='p',
                                     sig_thresh=0.05, epoch_markers=None, title=None,
                                     save_path=None, show=True, dpi=150):
    """Domain-by-domain comparison of the background x treatment interaction coefficient ACROSS
    epochs (default: acute / day1_late / followup) - a dumbbell/slope plot connecting all epoch
    points per domain with a thin grey line, not another lollipop-from-zero forest plot, since
    the question here is inherently a multi-condition comparison per domain rather than one
    value per domain. One panel per sex; one marker shape per epoch (see EPOCH_MARKERS); red/blue
    = sign, filled/hollow = significant/not (same conventions as the other forest-style plots).

    Args:
        epoch_mm: output of compute_epoch_interaction() (only the background_x_treatment term
            is used - filtered internally).
        epoch_labels: dict epoch -> legend text; defaults to EPOCH_LABELS.
        sig_col: 'p' (default - nominal; splitting into epochs cuts the already-small n per
            group, so p_fdr here is usually too conservative to show anything) or 'p_fdr'.

    Returns: (fig, axes)
    """
    mm = epoch_mm[epoch_mm.term == 'background_x_treatment']
    order = feature_order if feature_order is not None else list(dict.fromkeys(mm['domain']))
    markers = epoch_markers or EPOCH_MARKERS
    labels = epoch_labels or EPOCH_LABELS
    epochs = [e for e in epochs if e in set(mm['epoch'])]

    # NOT sharex: the acute/day1_late epochs run on ~1/3 the whole-session n (day 1 only, and
    # male's day-1 acute window is further thinned by a known missing recording block - see
    # KNOWN_BAD_BLOCKS in prep.py), so an occasional unstable, large-magnitude estimate in one
    # sex would otherwise flatten the other sex's panel onto the same axis range.
    fig, axes = plt.subplots(1, len(sexes), figsize=(6.8 * len(sexes), max(4, 0.5 * len(order))),
                              squeeze=False)
    axes = axes[0]

    for ax, sex in zip(axes, sexes):
        y = np.arange(len(order))
        sub_sex = mm[mm.sex == sex]
        piv = sub_sex.pivot(index='domain', columns='epoch', values='coef').reindex(order)
        sig_piv = (sub_sex.pivot(index='domain', columns='epoch', values=sig_col) < sig_thresh).reindex(order)

        for yi, dom in zip(y, order):
            if dom not in piv.index:
                continue
            vals = [piv.loc[dom, e] for e in epochs if e in piv.columns and not pd.isna(piv.loc[dom, e])]
            if len(vals) >= 2:
                ax.plot(vals, [yi] * len(vals), color=COLOR_MUTED, lw=1.2, zorder=1, alpha=0.6)

        for e in epochs:
            if e not in piv.columns:
                continue
            xvals = piv[e].values
            valid = ~pd.isna(xvals)
            colors = np.where(xvals >= 0, '#e34948', '#2a78d6')
            sig = sig_piv[e].values if e in sig_piv.columns else np.zeros(len(xvals), dtype=bool)
            m = markers.get(e, 'o')
            ax.scatter(xvals[valid & sig], y[valid & sig], color=colors[valid & sig], marker=m,
                       s=80, zorder=3, edgecolor='white', linewidth=0.8)
            ax.scatter(xvals[valid & ~sig], y[valid & ~sig], facecolor='white',
                       edgecolor=colors[valid & ~sig], marker=m, s=65, zorder=2, linewidth=1.4)

        ax.axvline(0, color=COLOR_MUTED, linewidth=1, linestyle='--', zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([_domain_label(f) for f in order], fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel('ELA × MDMA interaction coefficient', fontsize=11)
        ax.set_title(sex, fontsize=13)
        ax.tick_params(axis='y', length=0)
        sns.despine(ax=ax, left=True)
        ax.grid(axis='x', color=COLOR_MUTED, alpha=0.25, linewidth=0.5)

    handles = [plt.Line2D([0], [0], marker=markers.get(e, 'o'), color=COLOR_INK, linestyle='',
                          markersize=9, label=labels.get(e, e.capitalize())) for e in epochs]
    handles += [
        plt.Line2D([0], [0], marker='o', color='#e34948', linestyle='', markersize=9,
                   label=f'coef >= 0, {sig_col} < {sig_thresh}'),
        plt.Line2D([0], [0], marker='o', color='#2a78d6', linestyle='', markersize=9,
                   label=f'coef < 0, {sig_col} < {sig_thresh}'),
        plt.Line2D([0], [0], marker='o', markerfacecolor='white', markeredgecolor=COLOR_MUTED,
                   linestyle='', markersize=9, label='n.s.'),
    ]
    # ncol = len(handles) (single row) - matplotlib's default multi-row legend fill is
    # column-major, which reads as a scrambled order for a handle list built epoch-then-sig
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.08))
    if title:
        fig.suptitle(title, color=COLOR_INK, fontsize=14)
    plt.tight_layout(rect=(0, 0.09, 1, 1))

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes


EPOCH_BAR_COLOR = {'acute': COLOR_ACUTE, 'day1_late': COLOR_ACUTE, 'followup': COLOR_SUSTAINED}
EPOCH_BAR_HATCH = {'acute': '', 'day1_late': '///', 'followup': ''}


def plot_epoch_normalization_bars(epoch_norm, feature_order=None, sexes=('female', 'male'),
                                   epochs=EPOCH_ORDER, bar_colors=None, bar_hatches=None,
                                   bar_labels=None, min_baseline_dz=0.3, title=None,
                                   save_path=None, show=True, dpi=150):
    """Domain-by-domain normalization index (see compute_normalization_table()), compared ACROSS
    epochs (default: acute / day1_late / followup - pass epochs=DAYGROUP_ORDER for the simpler
    day1-vs-days2-3 split) as a grouped diverging bar chart - is any MDMA-specific ELA-CTRL gap
    narrowing concentrated on the single dosing day, or does it hold up / emerge in the
    drug-free follow-up days? One panel per sex; one bar per domain x epoch (dodged).

    Encoding: COLOR groups by timescale (default amber = Day 1, teal = 'followup'/Days 2-3), so
    the acute-vs-sustained contrast reads at a glance; HATCH can further subdivide same-colored
    bars (default: solid Day-1-0-4h vs hatched Day-1-4-12h, when using the finer 3-epoch split -
    with the simpler 2-group DAYGROUP_ORDER split, leave bar_hatches at its default of '' for
    all epochs since there's nothing left to subdivide). Sign (gap narrows vs widens) is read
    from bar DIRECTION only (right/left of zero), not color, freeing color for the timescale
    contrast - a different chart grammar again from the dumbbell (interaction) and volcano
    (general effect) views, matched to this being a signed-magnitude comparison where the
    time-course is the point, not the sign (already established: green/rust in
    compute_normalization_table()'s original whole-session view).

    STABILITY FLAG: normalization_index is a difference of two shrink terms, each anchored to
    that arm's OWN baseline |dz| (see compute_normalization_table()) - when either the MDMA-box
    or saline-box baseline gap is small, the index can swing to a large value that reflects
    box-level noise on a near-zero reference, not a real MDMA effect (e.g. a saline arm's
    baseline dz of -0.37 swinging to -2.25 post produces the same large "shrink" a real effect
    would, purely from having little baseline signal to begin with). Bars where
    min(|dz_baseline_MDMAbox|, |dz_baseline_salinebox|) < `min_baseline_dz` are rendered at
    reduced opacity and annotated with that minimum baseline |dz| directly, so the chart doesn't
    let an unstable ratio be the biggest, most eye-catching bar on the slide.

    Args:
        epoch_norm: output of compute_epoch_normalization() (needs dz_baseline_MDMAbox and
            dz_baseline_salinebox columns, both present by default).
        bar_colors / bar_hatches / bar_labels: dicts epoch -> hex color / hatch pattern / legend
            text; default to EPOCH_BAR_COLOR / EPOCH_BAR_HATCH / EPOCH_LABELS (pass
            bar_labels=DAYGROUP_LABELS, bar_colors=DAYGROUP_COLOR for the 2-group split).
        min_baseline_dz: bars below this in EITHER arm's baseline |dz| are faded + labeled with
            the offending value; pass None to disable the check entirely.

    Returns: (fig, axes)
    """
    order = feature_order if feature_order is not None else list(dict.fromkeys(epoch_norm['feature']))
    colors_map = bar_colors or EPOCH_BAR_COLOR
    # bar_hatches=None -> old 3-epoch default (hatch-within-Day-1); pass {} explicitly for a
    # flat/no-hatch look (e.g. the 2-group DAYGROUP_ORDER split, which has nothing left to
    # subdivide once color already carries the day1-vs-followup distinction)
    hatches = EPOCH_BAR_HATCH if bar_hatches is None else bar_hatches
    labels = bar_labels or EPOCH_LABELS
    epochs = [e for e in epochs if e in set(epoch_norm['epoch'])]
    n_e = len(epochs)
    bar_h = 0.75 / n_e

    epoch_norm = epoch_norm.copy()
    epoch_norm['_min_base_dz'] = epoch_norm[['dz_baseline_MDMAbox', 'dz_baseline_salinebox']].abs().min(axis=1)

    # NOT sharex: see plot_epoch_interaction_dumbbell() - the reduced, uneven per-epoch n makes
    # an occasional large-magnitude estimate in one sex flatten the other sex's panel otherwise.
    fig, axes = plt.subplots(1, len(sexes), figsize=(6.8 * len(sexes), max(4, 0.6 * len(order))),
                              squeeze=False)
    axes = axes[0]

    for ax, sex in zip(axes, sexes):
        y = np.arange(len(order))
        sub_sex = epoch_norm[epoch_norm.sex == sex]
        piv = sub_sex.pivot(index='feature', columns='epoch',
                             values='normalization_index').reindex(order)
        base_piv = sub_sex.pivot(index='feature', columns='epoch',
                                  values='_min_base_dz').reindex(order)

        for i, e in enumerate(epochs):
            if e not in piv.columns:
                continue
            offset = (i - (n_e - 1) / 2) * bar_h
            vals = piv[e].values
            base_dz = base_piv[e].values if e in base_piv.columns else np.full(len(vals), np.nan)
            unstable = (min_baseline_dz is not None) & (base_dz < min_baseline_dz)
            alphas = np.where(unstable, 0.32, 0.95)
            for yi, v, a in zip(y + offset, np.nan_to_num(vals), alphas):
                ax.barh(yi, v, height=bar_h * 0.92, color=colors_map.get(e, COLOR_MUTED),
                        alpha=a, edgecolor='white', linewidth=0.6, hatch=hatches.get(e, ''),
                        zorder=2)
            for yi, v, u, bd in zip(y + offset, vals, unstable, base_dz):
                if u and not np.isnan(v):
                    xoff = 6 if v >= 0 else -6
                    ha = 'left' if v >= 0 else 'right'
                    ax.annotate(f'dz₀={bd:.2f}', (v, yi), xytext=(xoff, 0),
                                textcoords='offset points', fontsize=7, va='center', ha=ha,
                                color=COLOR_MUTED, style='italic')

        ax.axvline(0, color=COLOR_MUTED, linewidth=1, linestyle='--', zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([_domain_label(f) for f in order], fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel('Normalization index (MDMA − saline)\n← gap widens · gap narrows →',
                      fontsize=10.5)
        ax.set_title(sex, fontsize=13)
        ax.tick_params(axis='y', length=0)
        sns.despine(ax=ax, left=True)
        ax.grid(axis='x', color=COLOR_MUTED, alpha=0.25, linewidth=0.5)

    handles = [
        Patch(facecolor=colors_map.get(e, COLOR_MUTED), edgecolor='white',
              hatch=hatches.get(e, ''), label=labels.get(e, e.capitalize()))
        for e in epochs
    ]
    if min_baseline_dz is not None:
        handles.append(Patch(facecolor=COLOR_MUTED, alpha=0.32, edgecolor='white',
                              label=f'baseline |dz| < {min_baseline_dz} in either arm (unstable)'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.08))
    if title:
        fig.suptitle(title, color=COLOR_INK, fontsize=14)
    plt.tight_layout(rect=(0, 0.09, 1, 1))

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes


#%%
# ============================================================================
# STANDALONE SMOKE TEST - reproduces run_analysis.py steps 1-3 + 6-7 and exercises all three
# angles above, without saving. Only runs when this file is executed on its own
# (`python treatment_interaction.py`), not when imported.
# ============================================================================
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')

    from prep import load_data, filter_data
    from domain_scores import CORE_DOMAINS, CORE_DOMAIN_FEATURES, ALL_DOMAINS

    MOUSE_ID_COLS = ['mouse_ID', 'sex', 'age', 'background', 'treatment', 'box_ID']
    DATA_PATH = '../behavior_dataset/final/QC_table'

    df = load_data(female_path=f'{DATA_PATH}/female_12h_filtered.csv',
                    male_path=f'{DATA_PATH}/male_12h_filtered.csv')
    raw_features = sorted({f for spec in ALL_DOMAINS.values()
                            for f in spec['features'] + spec.get('flip', [])})

    active_all = filter_data(df, phase='active')
    mouse_sess = aggregate_to_mouse(active_all, MOUSE_ID_COLS + ['time_point'], raw_features)
    baseline_mask = mouse_sess['time_point'] == 'baseline'
    mouse_sess = compute_domain_scores(mouse_sess, domain_defs=CORE_DOMAINS,
                                        reference_mask=baseline_mask)
    deltas = mouse_level_deltas(mouse_sess, features=CORE_DOMAIN_FEATURES, id_cols=MOUSE_ID_COLS,
                                 session_col='time_point', session_values=('baseline', 'MDMA'))

    plot_deltas_by_condition_grid(deltas, CORE_DOMAIN_FEATURES[:4],
                                   title='Treatment deltas by condition (smoke test)')

    mm = run_domain_mixedmodels(deltas, CORE_DOMAIN_FEATURES)
    print(mm[mm.term == 'background_x_treatment'].sort_values('p').to_string(index=False))
    plot_mixedmodel_forest(mm, feature_order=CORE_DOMAIN_FEATURES,
                            title='Background x treatment interaction (smoke test)')
    plot_bg_x_treatment_interaction(deltas, mm, sex='male')

    norm = compute_normalization_table(mouse_sess, CORE_DOMAIN_FEATURES)
    print(norm.to_string(index=False))
    plot_normalization(norm, sex='male', feature_order=CORE_DOMAIN_FEATURES)

    gaps = compute_normalization_gaps(mouse_sess, CORE_DOMAIN_FEATURES[:4])
    plot_normalization_gap_grid(gaps, CORE_DOMAIN_FEATURES[:4],
                                 title='Does MDMA narrow the ELA-CTRL gap? (box-paired, smoke test)')
    plot_normalization_gap_grid(gaps, CORE_DOMAIN_FEATURES[:4], pair_sexes=True,
                                 title='Does MDMA narrow the ELA-CTRL gap? (pair_sexes, smoke test)')

    gaps_all = compute_normalization_gaps(mouse_sess, CORE_DOMAIN_FEATURES)
    plot_normalization_gap_grid(gaps_all, CORE_DOMAIN_FEATURES, pair_sexes=True, n_cols=4,
                                 title='Does MDMA narrow the ELA-CTRL gap? (pair_sexes, n_cols=4, '
                                       'smoke test)')

    df_3h = load_data(female_path=f'{DATA_PATH}/female_3h_filtered.csv',
                       male_path=f'{DATA_PATH}/male_3h_filtered.csv')
    window_deltas = compute_window_deltas(df_3h, domain_defs=CORE_DOMAINS, id_cols=MOUSE_ID_COLS)
    plot_delta_by_window_grid(window_deltas, CORE_DOMAIN_FEATURES[:4],
                               full_session_deltas=deltas,
                               title='Time-course of the MDMA effect (smoke test)')
