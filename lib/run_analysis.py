"""
NOTE: EVERYTHING WILL BE MOVED TO THE JUPYTER NOTEBOOK VERSION.
 
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
    7. Treatment effect x background: does ELA modulate the MDMA response (mixed model, 7.1),
       does MDMA move ELA animals toward the CTRL level (7.2), and what's the drug effect's
       time-course, small vs large windows (7.3)? See treatment_interaction.py.
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
from stats_utils import paired_effect_table, box_level_delta_table


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
savepath = "/Users/xiuqi.ji/Library/CloudStorage/OneDrive-KarolinskaInstitutet/MDMA_ELA/social_box_2026/plots/baseline_ELA_vs_CTRL"
suffix ='3days'
wide = box_paired_wide(mouse_baseline, features=CORE_DOMAIN_FEATURES)
# print result table, uncomment to_csv line to save
print("=" * 70)
print("BASELINE: ELA vs CTRL, domain scores (box-paired)")
print("=" * 70)
baseline_effect_tables = {}
for sex in ['female', 'male']:
    sub = wide[wide.sex == sex]
    res = paired_effect_table(sub, CORE_DOMAIN_FEATURES)
    #res.to_csv(os.path.join(savepath, f'baseline_effect_table_{suffix}_{sex}_core.csv'),index=False)
    baseline_effect_tables[sex] = res
    print(f"\n--- {sex} (n={len(sub)} box pairs) ---")
    print(res[['feature', 'cohen_dz', 'p_ttest', 'p_fdr']].to_string(index=False))
#%% PLT
# Forest plot of the table above (see plot_effect_table.py for the function signature).
# Pass save_path=None (default) to just display; set it to write a PNG instead/as well.
from plot_effect_table import plot_effect_table

plot_effect_table(
    baseline_effect_tables,
    title='BASELINE: ELA vs CTRL, domain scores (box-paired)',
    save_path=os.path.join(savepath,f'baseline_effect_all_features_{suffix}.png'),
)

#%%
# ============================================================================
# 4.1. BASELINE ELA vs CTRL details
# ============================================================================
# Box-level detail underneath the forest plot above: one line per box connecting its ELA value
# to its CTRL value, for the domains with the strongest baseline effect per sex (see
# plot_by_box.py - same paired_effect_table() data source as the forest plot, just not
# collapsed to a single dz per panel).
from plot_by_box import plot_domains_by_box_grid, top_panels_by_effect
#%%
baseline_panels = top_panels_by_effect(baseline_effect_tables, n_per_sex=3)
plot_domains_by_box_grid(
    wide, baseline_panels,
    title='BASELINE: ELA vs CTRL, box-level detail (top hits per sex)',
    save_path=os.path.join(savepath, f'baseline_box_detail_{suffix}.png'),
)
#%%
custom_panels = [
    ('social_hierarchy_score', 'male'),
    ('social_hierarchy_score', 'female'),
    ('locomotion_score', 'male'),
    ('locomotion_score', 'female'),
    ('exploration_score', 'male'),
    ('exploration_score', 'female'),
]
plot_domains_by_box_grid(
    wide, custom_panels,
    title='BASELINE: ELA vs CTRL, box-level detail (top hits per sex)',
    save_path=os.path.join(savepath, f'baseline_box_detail_{suffix}_1.png'),
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

# Day-on-x-axis trajectory plot (mean+/-SEM per day, box-paired, with per-day Cohen's dz
# annotations) - see plot_domain_score_by_day.py for the function signature.
from plot_domain_score_by_day import plot_domain_score_by_day, plot_domain_scores_by_day_grid

plot_domain_scores_by_day_grid(
    wide_by_day,
    CORE_DOMAIN_FEATURES,
    title='BASELINE: ELA vs CTRL, domain scores by day (box-paired)',
    save_path=os.path.join(savepath, f'baseline_domain_scores_by_day_{suffix}.png'),
)

#%%
# ============================================================================
# 6. EXPLORE BY TIME WINDOW (needs a finer-resolution file)
# ============================================================================
# temporal_dynamics.py generalizes this: any resolution (1h/2h/3h/4h/6h/12h), any domain score OR
# raw feature, split by background (2-way, this step) or by condition (4-way, see step 7 below).
# Ages (P35/P42) are pooled, not split - see temporal_dynamics.py module docstring.
from temporal_dynamics import (
    load_temporal_data, plot_temporal_grid, plot_session_trajectory_grid,
)

RESOLUTION = '6h'  # swap for '1h' / '2h' / '4h' / '6h' / '12h' as desired
baseline_by_window = load_temporal_data(RESOLUTION, domain_defs=CORE_DOMAINS,
                                         time_points='baseline', data_path=DATA_PATH,
                                         keep_day=True)

plot_temporal_grid(
    baseline_by_window, CORE_DOMAIN_FEATURES, group='background',
    ylabel='z-scored domain composite',
    title=f'BASELINE: ELA vs CTRL, domain scores across the active phase ({RESOLUTION} bins)',
    save_path=os.path.join(savepath, f'baseline_temporal_{RESOLUTION}.png'),
)

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
#%%
# ----------------------------------------------------------------------------
# 7.0. Raw look at the deltas, before fitting anything: one panel per core domain x sex,
#      individual-mouse delta in each of the 4 background x treatment conditions
#      (CTRL_saline, CTRL_MDMA, ELA_saline, ELA_MDMA) - see treatment_interaction.py.
# ----------------------------------------------------------------------------
from treatment_interaction import (
    plot_deltas_by_condition_grid,
    run_domain_mixedmodels, plot_bg_x_treatment_interaction, plot_mixedmodel_forest,
    compute_normalization_table, plot_normalization,
    compute_window_deltas, plot_delta_by_window_grid,
)

savepath_tx = "/Users/xiuqi.ji/Library/CloudStorage/OneDrive-KarolinskaInstitutet/MDMA_ELA/social_box_2026/plots/treatment_MDMA_vs_saline"
#%%
plot_deltas_by_condition_grid(
    deltas, CORE_DOMAIN_FEATURES, sexes=('female', 'male'),
    title='TREATMENT: raw deltas by condition (core domains, before model fit)',
    save_path=os.path.join(savepath_tx, 'treatment_deltas_by_condition.png'),
)

#%%
# ----------------------------------------------------------------------------
# 7.1. Does ELA modulate the MDMA response? Mixed model: delta ~ background * treatment,
#      random intercept = box. One fit per domain per sex (see treatment_interaction.py).
# ----------------------------------------------------------------------------
mm = run_domain_mixedmodels(deltas, CORE_DOMAIN_FEATURES, sexes=('female', 'male'))
print("\n" + "=" * 70)
print("MIXED MODEL: delta ~ background * treatment (random=box), all core domains")
print("(nominal p; p_fdr corrects across the 12 domains within each sex x term)")
print("=" * 70)
print(mm[mm.term == 'background_x_treatment'].sort_values('p')
      [['domain', 'sex', 'coef', 'p', 'p_fdr', 'n']].to_string(index=False))

# Overview: which domains show a background x treatment interaction at all (both sexes side by
# side) - look at this before drilling into individual-domain panels below.
plot_mixedmodel_forest(
    mm, term='background_x_treatment', feature_order=CORE_DOMAIN_FEATURES,
    title='Background x treatment interaction, by domain (does ELA modulate the MDMA response?)',
    save_path=os.path.join(savepath_tx, 'treatment_mixedmodel_interaction_forest.png'),
)

# Drill-down: the domains with the strongest interaction, per sex, with individual mice shown
# (faint points) so within-group response variability is visible, not just the mean.
for sex in ['female', 'male']:
    plot_bg_x_treatment_interaction(
        deltas, mm, sex=sex, n_panels=12,
        save_path=os.path.join(savepath_tx, f'treatment_bg_x_interaction_{sex}_all.png'),
    )

#%%
# ----------------------------------------------------------------------------
# 7.2. Does MDMA bring ELA animals closer to the CTRL level? ("normalization" / potential
#      therapeutic direction.) Compares the box-paired ELA-CTRL gap (Cohen's dz) at baseline vs.
#      post-injection, separately in MDMA-dosed and saline-dosed boxes - a domain counts as
#      "normalizing" if the gap narrows specifically in MDMA-dosed boxes, beyond whatever
#      narrowing saline-dosed boxes show on retest alone (normalization_index > 0).
# ----------------------------------------------------------------------------
norm_table = compute_normalization_table(mouse_sess, CORE_DOMAIN_FEATURES, sexes=('female', 'male'))
print("\n" + "=" * 70)
print("NORMALIZATION: does MDMA narrow the ELA-CTRL gap more than saline (retest) does?")
print("=" * 70)
print(norm_table[['feature', 'sex', 'dz_baseline_MDMAbox', 'dz_post_MDMAbox',
                   'shrink_MDMA', 'shrink_saline', 'normalization_index']]
      .sort_values('normalization_index', ascending=False).to_string(index=False))

for sex in ['female', 'male']:
    plot_normalization(
        norm_table, sex=sex, feature_order=CORE_DOMAIN_FEATURES,
        save_path=os.path.join(savepath_tx, f'treatment_normalization_{sex}.png'),
    )

#%%
# ----------------------------------------------------------------------------
# 7.3. Time-course of the drug effect: small (RESOLUTION-bin, from step 6) windows vs. the
#      whole-session (12h, "large window") number already computed above. Restricted to the
#      domains with the clearest overall MDMA main effect, to keep the grid readable - swap in
#      any CORE_DOMAIN_FEATURES subset you want to look at instead.
# ----------------------------------------------------------------------------
df_fine = load_data(female_path=os.path.join(DATA_PATH, f'female_{RESOLUTION}_filtered.csv'),
                     male_path=os.path.join(DATA_PATH, f'male_{RESOLUTION}_filtered.csv'))
window_deltas = compute_window_deltas(df_fine, domain_defs=CORE_DOMAINS, id_cols=MOUSE_ID_COLS)

top_mdma_domains = (mm[mm.term == 'treatment_MDMA'].groupby('domain')['p'].min()
                     .sort_values().head(4).index.tolist())

plot_delta_by_window_grid(
    window_deltas, top_mdma_domains, full_session_deltas=deltas,
    title=f'Time-course of the MDMA effect ({RESOLUTION} bins; dashed = whole-session mean, '
          'dotted = saline-box reference)',
    save_path=os.path.join(savepath_tx, 'treatment_delta_by_window.png'),
)

#%%
# ----------------------------------------------------------------------------
# 7.3.1. Same time-course, but the ABSOLUTE value (z-scored composite) rather than the
#        MDMA-vs-baseline delta above, and the FULL arc - baseline days 1-3, a visual break at
#        the treatment boundary, then post-injection days 1-3 - in one panel per domain, split by
#        condition (CTRL_saline/CTRL_MDMA/ELA_saline/ELA_MDMA, since both treatment arms are in
#        play here). Companion view to 7.3 (which shows how far each group MOVED); this shows
#        what each group's whole trajectory actually looks like, not just the post-injection
#        session in isolation.
#        Loads BOTH time_points with keep_day=True (day kept as its own axis, not averaged) so
#        the z-scoring reference is baseline (same scale as 7.1-7.3) AND each day's shape within
#        each session is visible - see temporal_dynamics.plot_session_trajectory().
#        All core domains, wrapped into a grid, one figure per sex (12 domains is too wide for a
#        single row - see plot_session_trajectory_grid()'s docstring).
# ----------------------------------------------------------------------------
full_traj_by_window = load_temporal_data(RESOLUTION, domain_defs=CORE_DOMAINS,
                                          time_points=['baseline', 'MDMA'], data_path=DATA_PATH,
                                          keep_day=True)

for sex in ['female', 'male']:
    plot_session_trajectory_grid(
        full_traj_by_window, CORE_DOMAIN_FEATURES, sex=sex, group='condition', ncols=3,
        title=f'{sex}: BASELINE -> POST-INJECTION domain-score trajectory by condition '
              f'({RESOLUTION} bins)',
        save_path=os.path.join(savepath_tx, f'treatment_session_trajectory_{sex}_{RESOLUTION}.png'),
    )

#%%
# ----------------------------------------------------------------------------
# 7.4. Post-injection box-level detail: the same box_by_box view as step 4.1, but for the
#      post-injection session, with MDMA-dosed vs. saline-dosed boxes colored separately within
#      each panel (instead of pooling treatment like the baseline plot does) - the individual-
#      box view underneath the interaction plots in 7.1.
# ----------------------------------------------------------------------------
mdma_session = mouse_sess[mouse_sess.time_point == 'MDMA']
wide_post = box_paired_wide(mdma_session, features=CORE_DOMAIN_FEATURES,
                             extra_group_cols=('sex', 'treatment'))

post_panels = [(f, sex) for sex in ['male', 'female']
               for f in mm[(mm.sex == sex) & (mm.term == 'background_x_treatment')]
               .sort_values('p')['domain'].head(3)]
custom_panels = [
    ('social_hierarchy_score', 'male'),
    ('social_hierarchy_score', 'female'),
    ('locomotion_score', 'male'),
    ('locomotion_score', 'female'),
    ('exploration_score', 'male'),
    ('exploration_score', 'female'),
]

plot_domains_by_box_grid(
    wide_post, custom_panels, group_col='treatment',
    title='POST-INJECTION: ELA vs CTRL, box-level detail, split by treatment',
    save_path=os.path.join(savepath_tx, 'treatment_post_injection_box_detail_1.png'),
)

#%%
# ----------------------------------------------------------------------------
# 7.5. Same box-level detail, but on the delta (post-injection - baseline) rather than the raw
#      post-injection score - i.e. box_paired_wide() applied to `deltas` (step 7's mouse-level
#      delta table) instead of to mouse_sess. Same domains/panels as 7.4 for a direct
#      side-by-side: 7.4 shows where each box ends UP, 7.5 shows how far each box MOVED.
# ----------------------------------------------------------------------------
delta_features = [f'{f}__delta' for f in CORE_DOMAIN_FEATURES]
wide_delta = box_paired_wide(deltas, features=delta_features, extra_group_cols=('sex', 'treatment'))

# post_panels are plain domain names (e.g. 'speeding_score') - value_suffix='__delta' below
# appends '__delta' to build the actual column lookup ('speeding_score__delta_ELA', etc.),
# same domains/panel selection as 7.4.
plot_domains_by_box_grid(
    wide_delta, post_panels, group_col='treatment', value_suffix='__delta',
    ylabel='Δ (post-injection − baseline)\nz-scored composite',
    title='POST-INJECTION: ELA vs CTRL, box-level Δ detail, split by treatment',
    save_path=os.path.join(savepath_tx, 'treatment_delta_box_detail.png'),
)

#%%
# ============================================================================
# 8. ADD MORE HERE
# ============================================================================
# Ideas already discussed that aren't wired up as reusable functions yet - feel free to build
# these out using the building blocks above:
#   - age-split analysis: filter_data(..., age='P35') vs age='P42', then repeat steps 2-4
#   - individual stratification / clustering (dominance x social quadrants, k-means on the
#     full domain-score vector) - see chat history for example code, not yet in lib/
#   - matched-window recovery for the male P35 day-1 gap (see matched_window_recovery.py)
#   - treatment_interaction.py's compute_window_deltas() generalizes to 1h/2h/4h/6h files too -
#     useful if 3h bins turn out too coarse (or too noisy) for a domain of interest
