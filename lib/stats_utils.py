"""
stats_utils.py - generic statistical helpers. Not tied to any specific domain or dataset shape.
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf


def paired_effect_table(wide_box_df, features, group_a_suffix='_ELA', group_b_suffix='_CTRL'):
    """Paired (by box) comparison of feature_A vs feature_B across boxes. Returns a tidy table:
    feature, n_pairs, mean_A, mean_B, mean_diff (A-B), cohen_dz (paired effect size), t, p,
    p_wilcoxon, p_fdr. Input: output of aggregate.box_paired_wide()."""
    rows = []
    for feat in features:
        col_a, col_b = f'{feat}{group_a_suffix}', f'{feat}{group_b_suffix}'
        if col_a not in wide_box_df.columns or col_b not in wide_box_df.columns:
            continue
        a, b = wide_box_df[col_a], wide_box_df[col_b]
        mask = a.notna() & b.notna()
        a, b = a[mask], b[mask]
        n = len(a)
        if n < 3:
            continue
        diff = a.values - b.values
        d_mean, d_sd = diff.mean(), diff.std(ddof=1)
        dz = d_mean / d_sd if d_sd > 0 else np.nan
        t, p = stats.ttest_rel(a, b)
        try:
            _, w_p = stats.wilcoxon(a, b)
        except ValueError:
            w_p = np.nan
        rows.append(dict(feature=feat, n_pairs=n, mean_A=a.mean(), mean_B=b.mean(),
                          mean_diff=d_mean, cohen_dz=dz, t=t, p_ttest=p, p_wilcoxon=w_p))
    res = pd.DataFrame(rows)
    if len(res):
        res['p_fdr'] = multipletests(res['p_ttest'], method='fdr_bh')[1]
        res = res.sort_values('cohen_dz', key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return res


def box_level_delta_table(delta_df, features, group_col='treatment',
                           group_a='MDMA', group_b='saline', box_col='box_ID', avg_cols=('sex',)):
    """Unpaired (independent boxes) comparison of a delta column between two groups (e.g.
    MDMA-dosed vs saline-dosed boxes). Averages mice within box first. Input: output of
    aggregate.mouse_level_deltas() (columns named '<feature>__delta')."""
    box_means = delta_df.groupby(list(avg_cols) + [box_col, group_col], as_index=False)[
        [f'{f}__delta' for f in features]].mean()
    rows = []
    for feat in features:
        col = f'{feat}__delta'
        a = box_means.loc[box_means[group_col] == group_a, col].dropna()
        b = box_means.loc[box_means[group_col] == group_b, col].dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        pooled_sd = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
        hedges_g = (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else np.nan
        t, p = stats.ttest_ind(a, b, equal_var=False)
        try:
            _, u_p = stats.mannwhitneyu(a, b, alternative='two-sided')
        except ValueError:
            u_p = np.nan
        rows.append(dict(feature=feat, n_A=len(a), n_B=len(b), mean_delta_A=a.mean(),
                          mean_delta_B=b.mean(), hedges_g=hedges_g, t=t, p_ttest=p, p_mannwhitney=u_p))
    res = pd.DataFrame(rows)
    if len(res):
        res['p_fdr'] = multipletests(res['p_ttest'], method='fdr_bh')[1]
        res = res.sort_values('hedges_g', key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return res


def run_mixedlm(df, y_col, formula_rhs, group_col):
    """Thin wrapper around statsmodels MixedLM so callers don't repeat the same formula-string
    boilerplate. Returns the fitted result object (access .params, .pvalues, .summary()).

    Example:
        fit = run_mixedlm(delta_df, 'speeding_score__delta',
                           "C(background, Treatment('CTRL')) * C(treatment, Treatment('saline'))",
                           group_col='box_ID')
        print(fit.pvalues)
    """
    d = df[[group_col, y_col]].copy()
    # pull in any columns referenced in the formula (crude but sufficient: anything that looks
    # like an identifier and matches a df column)
    import re
    for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula_rhs):
        if tok in df.columns and tok not in d.columns:
            d[tok] = df[tok]
    d = d.dropna()
    formula = f"{y_col} ~ {formula_rhs}"
    md = smf.mixedlm(formula, d, groups=d[group_col])
    return md.fit(reml=True)
