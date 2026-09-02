"""
domain_scores.py - domain composite score definitions and computation.

Base structure is v3 (the version reviewed and approved), with two targeted changes made on
top of it:

  1. Integrated previously-unused mean_duration ("bout length") features as new SUPPLEMENTARY
     domains, in the same spirit as nest_fragmentation_score - each is checked against its
     matching frequency/amount feature and added only where it's not redundant.

  2. Dropped agonistic_engagement_score (chasing_duration_fraction + chased_duration_fraction
     averaged together). It didn't make sense: for the male ELA effect, chasing (dz=-1.41) and
     being chased (dz=+1.34) move in OPPOSITE directions, so averaging their z-scores produced a
     partially-cancelled, hard-to-interpret dz=-0.84 that didn't cleanly mean anything.
     social_hierarchy_score (normDS + chasing_duration_ratio) is retained unchanged and already
     captures "who wins". Separately, chasing_activity_score and being_chased_score ARE added
     as CORE domains (not as a merged pair) because they show strong, clean, OPPOSITE-signed
     effects on their own - i.e. they tell you something real and different from each other,
     just not something that should be blended into one number.
     The same opposite-direction pattern was found for chasing_mean_duration / chased_mean_
     duration (dz=-0.85 vs +0.88, male) - kept as two separate supplementary bout-length scores
     for the same reason.

Everything else (feature membership, flip conventions, robust z-scoring) is unchanged from v3.
"""
import numpy as np
import pandas as pd
from scipy.stats import skew as skew_fn

SKEW_THRESHOLD = 1.0

# ============================================================================
# CORE domains - primary hypothesis-testing set (v3 base, minus agonistic_engagement_score,
# plus chasing_activity_score / being_chased_score)
# ============================================================================
CORE_DOMAINS = {
    'nest_occupancy_score': {
        'features': ['nest_fraction'], 'flip': [],
        'definition': 'Time in nest relative to total observed time. During the ACTIVE phase, '
                       'elevated nest occupancy plausibly reflects withdrawal / reduced '
                       'environmental engagement rather than normal sleep.',
    },
    'locomotion_score': {
        'features': ['total_distance', 'mean_speed', 'median_speed', 'mean_abs_acceleration'],
        'flip': [],
        'definition': 'General locomotor output/vigor - "how much and how fast" a mouse moves. '
                       'These four features form one tight empirical cluster (r=0.75-0.99).',
    },
    'movement_complexity_score': {
        'features': ['mean_abs_angular_velocity'], 'flip': [],
        'definition': 'Path tortuosity / turning rate. Near-orthogonal to the locomotion '
                       'cluster (r=-0.07 to -0.33) - a distinct construct (search strategy / '
                       'scanning behavior), not simply "moving more or faster".',
    },
    'inactivity_score': {
        'features': ['motionless_duration_fraction', 'motionless_event_rate'], 'flip': [],
        'definition': 'Time immobile OUTSIDE the nest (freezing/pausing in the open). '
                       'Ethologically distinct from nest_occupancy_score - immobility while '
                       '"on duty" in the open arena, which in rodent literature often indexes '
                       'vigilance/anxiety-like freezing rather than restorative rest.',
    },
    'speeding_score': {
        'features': ['speeding_duration_fraction', 'speeding_event_rate'], 'flip': [],
        'definition': 'Frequency/duration of discrete high-speed ("dash") bouts. NOT the same '
                       'construct as locomotion_score: mean_speed correlates NEGATIVELY with '
                       'speeding features (r=-0.51) - mice with high average pace are not the '
                       'same mice who dash a lot.',
    },
    'exploration_score': {
        'features': ['s_wall_duration_fraction', 's_wall_event_rate',
                     'ramp1_duration_fraction', 'ramp1_event_rate',
                     'ramp2_duration_fraction', 'ramp2_event_rate',
                     'non_wall_duration_fraction', 'non_wall_event_rate',
                     'woodstick_duration_fraction', 'woodstick_event_rate'],
        'flip': [],
        'definition': 'Time directed at enrichment objects (s-wall, 2 ramps, woodstick) plus '
                       'non_wall. Per data owner, non_wall is a center-arena rectangle away '
                       'from walls, while nest/feeders/water/ramps/woodstick are all wall-'
                       'adjacent - functionally close to a center-zone/thigmotaxis measure, '
                       'though not a bare open field (s-wall is also central), so treated as a '
                       'proxy only. Included here as in v3.',
    },
    'feeding_drinking_amount_score': {
        'features': ['feeder_prox_duration_fraction', 'feeder_prox_event_rate',
                     'feeder_dist_duration_fraction', 'feeder_dist_event_rate',
                     'water_prox_duration_fraction', 'water_prox_event_rate',
                     'water_dist_duration_fraction', 'water_dist_event_rate'],
        'flip': [],
        'definition': 'Overall ingestive-zone engagement, prox+dist combined (magnitude, not '
                       'location).',
    },
    'resource_proximity_preference_score': {
        'features': ['feeder_prox_pref', 'water_prox_pref'], 'flip': [],
        'definition': 'PROX zones sit closer to the nest than DIST zones, so the balance '
                       'between them is a spatial-preference/risk-tolerance measure, not an '
                       'amount measure. Confirmed near-orthogonal to feeding_drinking_amount_'
                       'score (r=0.12 feeder, r=-0.08 water). Feeder & water preference only '
                       'weakly correlate with each other (r=0.20) - use feeder_preference_score '
                       '/ water_preference_score (supplementary) for resource-specific detail.',
    },
    'social_affiliation_nest_score': {
        'features': ['weighted_co_occupancy'], 'flip': ['alone_fraction'],
        'definition': 'Huddling/resting togetherness in the nest specifically.',
    },
    'social_affiliation_activity_score': {
        'features': ['feeding_together_fraction', 'drinking_together_fraction',
                     'ramps_together_fraction', 's_wall_together_fraction'],
        'flip': [],
        'definition': 'Co-occupancy while engaged in non-resting activities. Confirmed near-'
                       'zero correlation with nest affiliation (r=-0.03) - resting togetherness '
                       'and active togetherness are empirically independent axes. (Per-activity '
                       '*_alone_fraction columns are exact complements of *_together_fraction, '
                       'r=-1.00 - using together_fraction alone already captures the full '
                       'information.)',
    },
    'social_hierarchy_score': {
        'features': ['normDS', 'chasing_duration_ratio'], 'flip': [],
        'definition': 'Dominance OUTCOME - "who wins" when a chase interaction happens. '
                       'chasing_event_ratio dropped (r=0.997 with chasing_duration_ratio - '
                       'redundant); chased_duration_ratio dropped (exact complement, r=-1.00). '
                       'Unchanged from v3.',
    },
    'chasing_activity_score': {
        'features': ['chasing_duration_fraction', 'chasing_event_rate'], 'flip': [],
        'definition': 'How often/long this mouse initiates chases of others - an assertive/'
                       'pursuit-role behavior. NOT part of social_hierarchy_score (which is '
                       'about outcome, not role-frequency). Kept SEPARATE from being_chased_'
                       'score rather than averaged into one "engagement" score: the two roles '
                       'moved in OPPOSITE directions for the male ELA effect (dz=-1.41 vs '
                       '+1.34) - averaging would produce a misleading, partially-cancelled '
                       'number.',
    },
    'being_chased_score': {
        'features': ['chased_duration_fraction', 'chased_event_rate'], 'flip': [],
        'definition': 'How often/long this mouse is the target of chases - a defensive/evasive-'
                       'role behavior, ethologically distinct from chasing_activity_score. '
                       'Together the two scores show e.g. that ELA males aren\'t just "losing '
                       'more" (lower social_hierarchy_score) but are specifically being '
                       'targeted MORE, not merely chasing less - information the dropped '
                       'agonistic_engagement_score obscured rather than revealed.',
    },
}

# ============================================================================
# SUPPLEMENTARY domains. Use for follow-up characterization, not as a first-pass FDR-corrected screen.
# ============================================================================
SUPPLEMENTARY_DOMAINS = {
    # --- v3 originals ---
    'social_affiliation_feeding_drinking_score': {
        'features': ['feeding_together_fraction', 'drinking_together_fraction'], 'flip': [],
        'definition': 'Finer split of social_affiliation_activity_score: co-occupancy '
                       'specifically during ingestive behavior. r=0.35 with the ROI version '
                       'below - related but not redundant.',
    },
    'social_affiliation_roi_score': {
        'features': ['ramps_together_fraction', 's_wall_together_fraction'], 'flip': [],
        'definition': 'Finer split of social_affiliation_activity_score: co-occupancy '
                       'specifically during object exploration.',
    },
    'feeder_proximity_preference_score': {
        'features': ['feeder_prox_pref'], 'flip': [],
        'definition': 'Resource-specific version of resource_proximity_preference_score.',
    },
    'water_proximity_preference_score': {
        'features': ['water_prox_pref'], 'flip': [],
        'definition': 'Resource-specific version of resource_proximity_preference_score.',
    },
    # --- mean_duration ("bout length") features ---
    'nest_fragmentation_score': {
        'features': ['nest_count'], 'flip': ['nest_mean_duration'],
        'definition': 'Bout structure independent of total nest time: many short visits '
                       '(high count, low mean_duration) vs. few long ones. r(count, mean_'
                       'duration)=-0.87 confirms these move opposite each other - combined as a '
                       'single "fragmentation" axis (high score = more, shorter visits = more '
                       'restless/vigilant nest-use pattern). nest_count and nest_mean_duration '
                       'were unused in v3; this integrates them.',
    },
    'inactivity_bout_length_score': {
        'features': ['motionless_mean_duration'], 'flip': [],
        'definition': 'Are freezing bouts brief pauses or sustained immobility? Distinct '
                       'from inactivity_score (total time/frequency, not bout length).',
    },
    'speeding_bout_length_score': {
        'features': ['speeding_mean_duration'], 'flip': [],
        'definition': 'Are dashes brief flinches or sustained sprints? Distinct from '
                       'speeding_score (total time/frequency, not bout length).',
    },
    'exploration_depth_score': {
        'features': ['s_wall_mean_duration', 'ramp1_mean_duration', 'ramp2_mean_duration',
                     'non_wall_mean_duration', 'woodstick_mean_duration'],
        'flip': [],
        'definition': 'Average bout length per visit to an exploration zone - '
                       '"thoroughness" of investigation, distinct from how often visited '
                       '(exploration_score). Note: showed the OPPOSITE direction from '
                       'exploration_score in males (dz=+0.86 vs -0.78) - i.e. fewer visits but '
                       'longer once there - worth keeping in mind as a frequency/depth '
                       'dissociation, not just a redundant confirmation of exploration_score.',
    },
    'feeding_drinking_depth_score': {
        'features': ['feeder_prox_mean_duration', 'feeder_dist_mean_duration',
                     'water_prox_mean_duration', 'water_dist_mean_duration'],
        'flip': [],
        'definition': 'Average bout length per visit to a feeding/drinking zone, distinct '
                       'from feeding_drinking_amount_score (total time/frequency). Unlike the '
                       'chasing/being-chased or exploration cases, these four do NOT show '
                       'opposite-signed effects (all same direction, though water shows a '
                       'clearer effect than feeder in this dataset) - safe to combine into one '
                       'composite; split into feeder- vs water-specific versions if you want '
                       'that resource-level detail.',
    },
    'chasing_bout_length_score': {
        'features': ['chasing_mean_duration'], 'flip': [],
        'definition': 'Average duration of a chase bout when THIS mouse is the pursuer. '
                       'Kept separate from chased_bout_length_score: chasing_mean_duration and '
                       'chased_mean_duration move in OPPOSITE directions for the male ELA '
                       'effect (dz=-0.85 vs +0.88), the same pattern as chasing_activity_score '
                       'vs being_chased_score - do not average these together.',
    },
    'chased_bout_length_score': {
        'features': ['chased_mean_duration'], 'flip': [],
        'definition': 'Average duration of a chase bout when THIS mouse is the target. '
                       'See chasing_bout_length_score for why this is kept separate.',
    },
}

ALL_DOMAINS = {**CORE_DOMAINS, **SUPPLEMENTARY_DOMAINS}
CORE_DOMAIN_FEATURES = list(CORE_DOMAINS.keys())
SUPPLEMENTARY_DOMAIN_FEATURES = list(SUPPLEMENTARY_DOMAINS.keys())
ALL_DOMAIN_FEATURES = list(ALL_DOMAINS.keys())


def robust_or_standard_z(x, ref_x):
    """z-score x against ref_x's distribution. Uses median/MAD (scaled to be consistent with
    SD under normality) if ref_x is notably skewed, else ordinary mean/SD."""
    ref_x = ref_x.dropna()
    if len(ref_x) < 5:
        return (x - ref_x.mean()) / ref_x.std() if ref_x.std() > 0 else np.nan
    sk = skew_fn(ref_x)
    if abs(sk) > SKEW_THRESHOLD:
        med = ref_x.median()
        mad = (ref_x - med).abs().median() * 1.4826
        return (x - med) / mad if mad > 0 else np.nan
    else:
        mu, sd = ref_x.mean(), ref_x.std()
        return (x - mu) / sd if sd > 0 else np.nan


def compute_domain_scores(df, domain_defs=None, reference_mask=None, group_col='sex'):
    """Add one z-scored composite column per domain in domain_defs to a copy of df.

    Z-scoring is done separately per `group_col` value (default 'sex'), using the mean/SD (or
    median/MAD if skewed) of `reference_mask` rows as the reference distribution - so pass
    reference_mask=(df['time_point']=='baseline') to keep baseline and MDMA-session values on
    the same scale, for instance. If reference_mask is None, uses all rows.

    domain_defs defaults to CORE_DOMAINS; pass ALL_DOMAINS to include supplementary splits,
    or any custom dict with the same {'features': [...], 'flip': [...]} structure to define
    your own domains without touching this file.
    """
    if domain_defs is None:
        domain_defs = CORE_DOMAINS
    out = df.copy()
    if reference_mask is None:
        reference_mask = pd.Series(True, index=out.index)
    for domain, spec in domain_defs.items():
        feats = spec['features']
        flips = set(spec.get('flip', []))
        z_cols = []
        for feat in feats + list(flips):
            if feat not in out.columns:
                continue
            zc = f'__z_{feat}'
            for grp in out[group_col].unique():
                sel = (out[group_col] == grp)
                ref = sel & reference_mask
                z = robust_or_standard_z(out.loc[sel, feat], out.loc[ref, feat])
                out.loc[sel, zc] = (-z if feat in flips else z)
            z_cols.append(zc)
        if z_cols:
            out[domain] = out[z_cols].mean(axis=1, skipna=True)
            out.drop(columns=z_cols, inplace=True)
    return out
