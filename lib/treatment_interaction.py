"""
treatment_interaction.py - does MDMA affect ELA and CTRL animals differently, and what does
that difference look like?

Three angles, each a compute_* (tidy DataFrame) + plot_* (visualization) pair, plus a raw-data
EDA plot (plot_deltas_by_condition_grid) meant to be looked at BEFORE any of the three:

  0. Raw deltas by condition - plain box+strip plot of the per-mouse delta in each of the 4
     background x treatment conditions (CTRL_saline, CTRL_MDMA, ELA_saline, ELA_MDMA), one panel
     per domain x sex, no model fit involved - a sanity check on the data the mixed model below
     is about to summarize.
       plot: plot_deltas_by_condition_grid()
  1. Background x treatment interaction - does ELA blunt/amplify the drug effect, domain by
     domain? (this is the generalized, reusable version of the old fig5_interaction.py, which
     hardcoded 4 panels and read from two stale intermediate CSVs that no longer exist)
       compute: run_domain_mixedmodels()        plot: plot_bg_x_treatment_interaction()
                                                 plot: plot_mixedmodel_forest() (all-domain overview)
  2. Normalization - does MDMA move ELA animals TOWARD the CTRL baseline level (potential
     therapeutic direction), beyond whatever shift saline-dosed boxes show on retest alone?
       compute: compute_normalization_table()   plot: plot_normalization()
  3. Time-course - is the drug effect front-loaded (early post-injection hours) or does it
     accumulate/persist across the whole session? Needs a finer-resolution QC_table file.
       compute: compute_window_deltas()         plot: plot_delta_by_window_grid()

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
        run_domain_mixedmodels, plot_bg_x_treatment_interaction, plot_mixedmodel_forest,
        compute_normalization_table, plot_normalization,
        compute_window_deltas, plot_delta_by_window_grid,
    )
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

from aggregate import aggregate_to_mouse, box_paired_wide, mouse_level_deltas
from domain_scores import compute_domain_scores
from stats_utils import paired_effect_table, run_mixedlm

# Palette conventions consistent with plot_effect_table.py / plot_domain_score_by_day.py
# (dataviz skill diverging pair + the sex/background convention already used there).
COLOR_CTRL = '#898781'
COLOR_MUTED = '#898781'
COLOR_INK = '#0b0b0b'
SEX_ELA_COLORS = {'female': '#e34948', 'male': '#2a78d6'}
COLOR_MDMA_GROUP = '#8e44ad'
COLOR_SALINE_GROUP = '#898781'


def _domain_label(feature):
    return feature.replace('_score', '').replace('_', ' ').capitalize()


# ============================================================================
# 0. RAW DELTAS BY CONDITION (EDA, no model fit - look at this before 1.)
# ============================================================================
CONDITION_ORDER = ['CTRL_saline', 'CTRL_MDMA', 'ELA_saline', 'ELA_MDMA']


def _condition_palette(sex):
    ela_color = SEX_ELA_COLORS.get(sex, '#e34948')
    return {'CTRL_saline': COLOR_CTRL, 'CTRL_MDMA': COLOR_CTRL,
            'ELA_saline': ela_color, 'ELA_MDMA': ela_color}


def plot_deltas_by_condition_grid(deltas, features, sexes=('female', 'male'),
                                   title=None, save_path=None, show=True, dpi=150):
    """Raw-data look at the per-mouse delta (post-injection - baseline) in each of the 4
    background x treatment conditions, BEFORE fitting the delta ~ background * treatment mixed
    model in run_domain_mixedmodels() below - a sanity check on what that model is about to
    summarize (outliers, obviously non-normal spread, near-empty cells, etc.).

    One panel per domain (columns) x sex (rows): box + individual jittered points, x-axis =
    CTRL_saline, CTRL_MDMA, ELA_saline, ELA_MDMA. Saline-dosed boxes (whose "post-injection"
    session is really just a retest, not a drug challenge - see module docstring) are drawn at
    reduced alpha so the eye lands on the two MDMA columns first.

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
            ax.set_xticks(range(len(CONDITION_ORDER)))
            ax.set_xticklabels(['CTRL\nsaline', 'CTRL\nMDMA', 'ELA\nsaline', 'ELA\nMDMA'],
                               fontsize=7.5)
            ax.set_xlabel('')
            ax.set_title(_domain_label(feat) if row == 0 else '', fontsize=9.5)
            ax.set_ylabel(f'{sex}\nΔ (post − baseline)\nz-scored composite' if col == 0 else '',
                          fontsize=9)
            sns.despine(ax=ax)

    fig.suptitle(title or 'Treatment deltas by condition (raw data, before model fit)',
                 color=COLOR_INK, fontsize=13, fontweight='bold', y=1.02)
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


def plot_mixedmodel_forest(mm, term='background_x_treatment', sexes=('female', 'male'),
                            feature_order=None, sig_col='p', sig_thresh=0.05,
                            title=None, save_path=None, show=True, dpi=150):
    """All-domain overview forest plot of one mixed-model term (default: the background x
    treatment interaction), one panel per sex - the "which domains show this pattern at all"
    view to look at before drilling into plot_bg_x_treatment_interaction() for individual
    domains. Same visual language as plot_effect_table.py.

    Args:
        mm: output of run_domain_mixedmodels().
        term: one of 'background_ELA', 'treatment_MDMA', 'background_x_treatment'.
        feature_order: top-to-bottom domain order shared across panels; defaults to the order
            domains first appear in mm.
        sig_col: 'p' (nominal) or 'p_fdr' - which column decides filled-vs-hollow markers.

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
        sig = res[sig_col] < sig_thresh

        ax.hlines(y, 0, res['coef'], color=colors, linewidth=1.5, zorder=1)
        ax.scatter(res.loc[sig, 'coef'], y[sig], color=colors[sig], s=70, zorder=2)
        ax.scatter(res.loc[~sig, 'coef'], y[~sig], facecolor='white', edgecolor=colors[~sig],
                   linewidth=1.5, s=70, zorder=2)
        ax.axvline(0, color=COLOR_MUTED, linewidth=1, linestyle='--', zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([_domain_label(f) for f in res['domain']], fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel(f'{term} coefficient', fontsize=11)
        n_txt = res["n"].iloc[0] if len(res) else "-"
        ax.set_title(f'{sex} (n={n_txt} mice)', fontsize=13)
        ax.tick_params(axis='y', length=0)
        sns.despine(ax=ax, left=True)
        ax.grid(axis='x', color=COLOR_MUTED, alpha=0.25, linewidth=0.5)

    handles = [
        plt.Line2D([0], [0], marker='o', color='#e34948', linestyle='', markersize=10,
                   label=f'coef >= 0, {sig_col} < {sig_thresh}'),
        plt.Line2D([0], [0], marker='o', color='#2a78d6', linestyle='', markersize=10,
                   label=f'coef < 0, {sig_col} < {sig_thresh}'),
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


def plot_delta_by_window(window_deltas, feature, sex, ax=None, window_col='time_window',
                          full_session_deltas=None, active_start_hour=12):
    """One domain's MDMA-vs-baseline delta across fine time windows (solid, mean+/-SEM, one
    line per background), with the same-window saline-box trajectory as a faint dotted
    reference (non-specific session-to-session drift), and - if `full_session_deltas` is given -
    a dashed horizontal line at the whole-session (large-window) mean delta for comparison, so
    the small-window time-course and the large-window headline number sit in the same panel.

    Returns: ax
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3.6, 4.3))

    col = f'{feature}__delta'
    ela_color = SEX_ELA_COLORS.get(sex, '#e34948')

    sub_mdma = window_deltas[(window_deltas.sex == sex) & (window_deltas.treatment == 'MDMA')][
        ['background', window_col, col]].dropna()
    windows = sorted(sub_mdma[window_col].unique(), key=_window_sort_key)
    x = np.arange(len(windows))

    for bg, style in [('CTRL', dict(color=COLOR_CTRL, marker='o')),
                       ('ELA', dict(color=ela_color, marker='s'))]:
        g = sub_mdma[sub_mdma.background == bg]
        means = g.groupby(window_col)[col].mean()
        sems = g.groupby(window_col)[col].sem()
        y = [means.get(w, np.nan) for w in windows]
        e = [sems.get(w, np.nan) for w in windows]
        ax.errorbar(x, y, yerr=e, label=f'{bg} (MDMA)', capsize=3, lw=2, markersize=6,
                    zorder=3, **style)

    sub_sal = window_deltas[(window_deltas.sex == sex) & (window_deltas.treatment == 'saline')][
        ['background', window_col, col]].dropna()
    for bg, color in [('CTRL', COLOR_CTRL), ('ELA', ela_color)]:
        g = sub_sal[sub_sal.background == bg]
        means = g.groupby(window_col)[col].mean()
        y = [means.get(w, np.nan) for w in windows]
        ax.plot(x, y, ls=':', lw=1.3, alpha=0.5, color=color, zorder=1)

    if full_session_deltas is not None:
        sub_full = full_session_deltas[(full_session_deltas.sex == sex)
                                        & (full_session_deltas.treatment == 'MDMA')]
        for bg, color in [('CTRL', COLOR_CTRL), ('ELA', ela_color)]:
            v = sub_full.loc[sub_full.background == bg, col].dropna()
            if len(v):
                ax.axhline(v.mean(), color=color, lw=1, ls='--', alpha=0.5, zorder=0)

    ax.axhline(0, color=COLOR_MUTED, lw=0.5, ls=':')
    ax.set_xticks(x)
    ax.set_xticklabels([_window_label(w, active_start_hour) for w in windows], fontsize=8)
    ax.set_xlabel('Hours since active-phase start', fontsize=8.5)
    ax.set_title(_domain_label(feature), fontsize=10.5)
    sns.despine(ax=ax)
    return ax


def plot_delta_by_window_grid(window_deltas, features, sexes=('female', 'male'),
                               full_session_deltas=None, title=None, save_path=None,
                               show=True, dpi=150):
    """Reference grid: one row per sex, one column per feature, each panel via
    plot_delta_by_window(). Dotted lines = same-window saline-box reference; dashed horizontal
    lines (when full_session_deltas is given) = whole-session mean, for the small-vs-large
    window comparison.

    Returns: (fig, axes)
    """
    fig, axes = plt.subplots(len(sexes), len(features),
                              figsize=(2.8 * len(features), 3.9 * len(sexes)), squeeze=False)

    for row, sex in enumerate(sexes):
        for col, feat in enumerate(features):
            ax = axes[row, col]
            plot_delta_by_window(window_deltas, feat, sex, ax=ax,
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

    df_3h = load_data(female_path=f'{DATA_PATH}/female_3h_filtered.csv',
                       male_path=f'{DATA_PATH}/male_3h_filtered.csv')
    window_deltas = compute_window_deltas(df_3h, domain_defs=CORE_DOMAINS, id_cols=MOUSE_ID_COLS)
    plot_delta_by_window_grid(window_deltas, CORE_DOMAIN_FEATURES[:4],
                               full_session_deltas=deltas,
                               title='Time-course of the MDMA effect (smoke test)')
