"""
run_analysis.py - template/tutorial script showing the analysis flow end-to-end.

This is NOT meant to be a final pipeline that produces and saves every possible output -
it's a working template showing how the pieces (prep -> aggregate -> domain_scores ->
stats_utils) fit together, with each stage kept as a separate, inspectable variable so you
can modify or extend any step without touching the others. Copy sections of this into a
notebook, or keep editing this file directly - either works with this structure.

Flow:
    1. Load & subset data (with day/phase/time_point/sex/age filters)
    2. Aggregate to mouse-level (session means, or day-level, or whatever grouping you want)
    3. Compute domain scores
    4. Box-paired ELA vs CTRL comparison (baseline)
    5. [starter] explore by day
    6. [starter] explore by time window (needs a finer-resolution file)
    7. [starter] treatment effect x background (mixed model)
    8. <-- add more analyses here as you go
"""
#%%
import sys
sys.path.insert(0, 'lib')
import pandas as pd
import os

from prep import load_data, filter_data
from aggregate import aggregate_to_mouse, box_paired_wide, mouse_level_deltas, box_level_means
from domain_scores import compute_domain_scores, CORE_DOMAINS, ALL_DOMAINS, CORE_DOMAIN_FEATURES
from stats_utils import paired_effect_table, box_level_delta_table, run_mixedlm


MOUSE_ID_COLS = ['mouse_ID', 'sex', 'age', 'background', 'treatment', 'box_ID']

#%%
# ============================================================================
# 1. LOAD & SUBSET
# ============================================================================
# Adjust these two paths to wherever your QC_table CSVs live relative to this script.
DATA_PATH = '/Users/xiuqi.ji/Library/CloudStorage/OneDrive-KarolinskaInstitutet/MDMA_ELA/social_box_2026/github_repo/behavior_dataset/final/QC_table'
FEMALE_12H_PATH = '/Users/xiuqi.ji/Library/CloudStorage/OneDrive-KarolinskaInstitutet/MDMA_ELA/social_box_2026/github_repo/behavior_dataset/final/QC_table/female_12h_filtered.csv'
MALE_12H_PATH = '/Users/xiuqi.ji/Library/CloudStorage/OneDrive-KarolinskaInstitutet/MDMA_ELA/social_box_2026/github_repo/behavior_dataset/final/QC_table/male_12h_filtered.csv'

df = load_data(female_path=FEMALE_12H_PATH, male_path=MALE_12H_PATH)

# Baseline, active phase, all 3 days (the default "headline" subset):
baseline_active = filter_data(df, phase='active', time_point='baseline')

# To restrict to specific days instead, e.g. day 1 only:
baseline_active_day1 = filter_data(df, phase='active', time_point='baseline', days=[1])
# Or day 2 and 3 only:
baseline_active_day23 = filter_data(df, phase='active', time_point='baseline', days=[2, 3])

#%%
# ============================================================================
# 2. AGGREGATE TO MOUSE-LEVEL
# ============================================================================
# "Session means": collapse days 1-3 into a single baseline value per mouse. This is the
# standard denominator for domain-score z-scoring (see step 3) and for the box-paired
# comparison in step 4.
from domain_scores import ALL_DOMAIN_FEATURES  # noqa: E402 - raw features needed pre-domain-score
RAW_FEATURES_FOR_DOMAINS = sorted({f for spec in ALL_DOMAINS.values()
                                    for f in spec['features'] + spec.get('flip', [])})

mouse_baseline = aggregate_to_mouse(baseline_active, MOUSE_ID_COLS, RAW_FEATURES_FOR_DOMAINS)
mouse_baseline_day1 = aggregate_to_mouse(baseline_active_day1, MOUSE_ID_COLS, RAW_FEATURES_FOR_DOMAINS)
mouse_baseline_day23 = aggregate_to_mouse(baseline_active_day23, MOUSE_ID_COLS, RAW_FEATURES_FOR_DOMAINS)

#%%
# ============================================================================
# 3. COMPUTE DOMAIN SCORES
# ============================================================================
# reference_mask=None here because mouse_baseline is ALREADY baseline-only, so "all rows" IS
# the baseline reference distribution. If you're working with a df that mixes baseline and
# MDMA-session rows (e.g. for the treatment-effect flow in step 7), pass
# reference_mask=(df['time_point']=='baseline') so both sessions are z-scored on the SAME scale.
#mouse_baseline = compute_domain_scores(mouse_baseline, domain_defs=CORE_DOMAINS)
# Use domain_defs=ALL_DOMAINS instead to include the supplementary finer-grained splits.
mouse_baseline = compute_domain_scores(mouse_baseline, domain_defs=ALL_DOMAINS)
mouse_baseline_day1 = compute_domain_scores(mouse_baseline_day1, domain_defs=ALL_DOMAINS)
mouse_baseline_day23 = compute_domain_scores(mouse_baseline_day23, domain_defs=ALL_DOMAINS)
#%%
# ============================================================================
# 4. BOX-PAIRED ELA vs CTRL COMPARISON (BASELINE)
# ============================================================================
wide = box_paired_wide(mouse_baseline, features=ALL_DOMAIN_FEATURES)

print("=" * 70)
print("BASELINE: ELA vs CTRL, domain scores (box-paired)")
print("=" * 70)
baseline_effect_tables = {}
for sex in ['female', 'male']:
    sub = wide[wide.sex == sex]
    res = paired_effect_table(sub, ALL_DOMAIN_FEATURES)
    baseline_effect_tables[sex] = res
    print(f"\n--- {sex} (n={len(sub)} box pairs) ---")
    print(res[['feature', 'cohen_dz', 'p_ttest', 'p_fdr']].to_string(index=False))
#%% PLT
# Forest plot of the table above (see plot_effect_table.py for the function signature).
# Pass save_path=None (default) to just display; set it to write a PNG instead/as well.
from plot_effect_table import plot_effect_table
savepath = "/Users/xiuqi.ji/Library/CloudStorage/OneDrive-KarolinskaInstitutet/MDMA_ELA/social_box_2026/plots"
plot_effect_table(
    baseline_effect_tables,
    title='BASELINE: ELA vs CTRL, domain scores (box-paired)',
    save_path=os.path.join(savepath,'baseline_effect_table_3days.png'),
)

#%%
# ============================================================================
# 5. [STARTER] EXPLORE BY DAY
# ============================================================================
# Same idea as step 2-4, but keep 'day' in the grouping instead of collapsing it, so you get
# one domain-score value per mouse PER DAY rather than one per mouse overall.
mouse_by_day = aggregate_to_mouse(baseline_active, MOUSE_ID_COLS + ['day'], RAW_FEATURES_FOR_DOMAINS)
mouse_by_day = compute_domain_scores(mouse_by_day, domain_defs=CORE_DOMAINS)
wide_by_day = box_paired_wide(mouse_by_day, features=CORE_DOMAIN_FEATURES, extra_group_cols=('sex', 'day'))
# From here: loop over `day` and call paired_effect_table per day, or plot day-on-x-axis
# trajectories - see the chat history for example plotting code (fig8_day_level.py) if useful,
# or write your own against this table.

#%%
# ============================================================================
# 6. [STARTER] EXPLORE BY TIME WINDOW (needs a finer-resolution file)
# ============================================================================
# Uncomment and point at a finer-resolution QC_table file (1h/2h/3h/4h/6h - same schema as 12h,
# except normDS/rank don't exist below 12h resolution, so hierarchy-related domains will be
# missing those components automatically - compute_domain_scores() skips missing features).
#
df_3h = load_data(female_path=os.path.join(DATA_PATH,'female_3h_filtered.csv'), 
                  male_path=os.path.join(DATA_PATH,'male_3h_filtered.csv'))
baseline_3h = filter_data(df_3h, phase='active', time_point='baseline')
mouse_by_window = aggregate_to_mouse(baseline_3h, MOUSE_ID_COLS + ['time_window'],
                                      RAW_FEATURES_FOR_DOMAINS)
mouse_by_window = compute_domain_scores(mouse_by_window, domain_defs=CORE_DOMAINS)
# -> group by time_window x background x sex and plot a time-course (mean +/- SEM per bin).

#%%
# ============================================================================
# 7. [STARTER] TREATMENT EFFECT x BACKGROUND
# ============================================================================
# Needs both baseline AND MDMA-session rows on the SAME z-score scale, hence reference_mask.
active_all = filter_data(df, phase='active')  # both time_points, active phase only
mouse_sess = aggregate_to_mouse(active_all, MOUSE_ID_COLS + ['time_point'], RAW_FEATURES_FOR_DOMAINS)
baseline_mask = mouse_sess['time_point'] == 'baseline'
mouse_sess = compute_domain_scores(mouse_sess, domain_defs=CORE_DOMAINS, reference_mask=baseline_mask)

deltas = mouse_level_deltas(mouse_sess, features=CORE_DOMAIN_FEATURES, id_cols=MOUSE_ID_COLS,
                             session_col='time_point', session_values=('baseline', 'MDMA'))

print("\n" + "=" * 70)
print("TREATMENT: box-level MDMA vs saline delta (core domains, both sexes pooled per test)")
print("=" * 70)
for sex in ['female', 'male']:
    sub = deltas[deltas.sex == sex]
    res = box_level_delta_table(sub, CORE_DOMAIN_FEATURES)
    print(f"\n--- {sex} ---")
    print(res[['feature', 'hedges_g', 'p_ttest', 'p_fdr']].to_string(index=False))

# Mixed model example: does ELA modulate the MDMA response, for one domain?
example_domain = 'speeding_score'
for sex in ['male']:
    sub = deltas[deltas.sex == sex]
    fit = run_mixedlm(sub, y_col=f'{example_domain}__delta',
                       formula_rhs="C(background, Treatment('CTRL')) * C(treatment, Treatment('saline'))",
                       group_col='box_ID')
    print(f"\n--- mixed model: {example_domain} delta ~ background * treatment ({sex}) ---")
    print(fit.pvalues)

# ============================================================================
# 8. ADD MORE HERE
# ============================================================================
# Ideas already discussed that aren't wired up as reusable functions yet - feel free to build
# these out using the building blocks above:
#   - age-split analysis: filter_data(..., age='P35') vs age='P42', then repeat steps 2-4
#   - individual stratification / clustering (dominance x social quadrants, k-means on the
#     full domain-score vector) - see chat history for example code, not yet in lib/
#   - matched-window recovery for the male P35 day-1 gap (see matched_window_recovery.py)
