"""
Recover the male_P35, day-1, active-phase, MDMA-session comparison using the
1-hour resolution file, instead of dropping the block entirely.

Ground truth (confirmed from male_1h_filtered.csv):
  - Missing: hours 0-5 of the active phase (time_window 12-13 .. 16-17), 5 hours, ZERO rows
    (not NaN - the rows don't exist at all for this specific block).
  - Recorded: hours 5-12 of the active phase (time_window 17-18 .. 23-24), 7 hours, present.

Strategy ("matched window", not imputation):
  - Recompute day-1 MDMA-session values from the 7 REAL recorded hours only (raw sums).
  - Recompute day-1 BASELINE-session values from the SAME 7-hour window (not the full 12h),
    so the comparison is apples-to-apples in duration.
  - All *_duration_fraction / *_event_rate / *_mean_duration are re-derived from raw
    duration/count sums over the matched window (confirmed formulas: fraction = duration /
    window outside_nest_duration; rate = count / window outside_nest_duration;
    mean_duration = duration / count).
  - normDS / rank are NOT hour-resolved in the source data in a way that can be safely
    re-aggregated (dominance is computed over the whole session, not additively over hours),
    so they are left as NaN for this window and simply excluded from this specific analysis.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd

RECORDED_WINDOWS = ['17-18','18-19','19-20','20-21','21-22','22-23','23-24']

# raw duration & count columns needed to rebuild every fraction/rate/mean_duration feature
RAW_DURATION_COLS = ['s_wall_duration','ramp1_duration','ramp2_duration','non_wall_duration',
                      'woodstick_duration','feeder_prox_duration','feeder_dist_duration',
                      'water_prox_duration','water_dist_duration','motionless_duration',
                      'speeding_duration','nest_duration','outside_nest_duration']
RAW_COUNT_COLS = ['s_wall_count','ramp1_count','ramp2_count','non_wall_count','woodstick_count',
                   'feeder_prox_count','feeder_dist_count','water_prox_count','water_dist_count',
                   'motionless_count','speeding_count','nest_count']
LOCOMOTION_RAW = ['total_distance','valid_locomotion_duration']  # summed; speed/accel need weighting
SOCIAL_FRAMES = ['nest_frames','weighted_sum','alone_sum','feeding_frames','drinking_frames',
                  'ramps_frames','s_wall_frames','feeding_together_frames','drinking_together_frames',
                  'ramps_together_frames','s_wall_together_frames','feeding_alone_frames',
                  'drinking_alone_frames','ramps_alone_frames','s_wall_alone_frames']

ROI_NAMES = ['s_wall','ramp1','ramp2','non_wall','woodstick',
             'feeder_prox','feeder_dist','water_prox','water_dist']


def build_matched_window_table(hourly_df, mouse_id_col='mouse_ID'):
    """Sum raw duration/count/frame columns over RECORDED_WINDOWS for each mouse,
    then re-derive fraction/rate/mean_duration features from the sums."""
    sub = hourly_df[hourly_df['time_window'].isin(RECORDED_WINDOWS)].copy()
    sum_cols = RAW_DURATION_COLS + RAW_COUNT_COLS + LOCOMOTION_RAW + SOCIAL_FRAMES
    sum_cols = [c for c in sum_cols if c in sub.columns]
    agg = sub.groupby(mouse_id_col)[sum_cols].sum()
    meta = sub.groupby(mouse_id_col)[['sex','age','background','treatment','condition','box_ID']].first()
    out = pd.concat([meta, agg], axis=1)

    for roi in ROI_NAMES:
        out[f'{roi}_duration_fraction'] = out[f'{roi}_duration'] / out['outside_nest_duration']
        out[f'{roi}_event_rate'] = out[f'{roi}_count'] / out['outside_nest_duration']
    out['motionless_duration_fraction'] = out['motionless_duration'] / out['outside_nest_duration']
    out['motionless_event_rate'] = out['motionless_count'] / out['outside_nest_duration']
    out['speeding_duration_fraction'] = out['speeding_duration'] / out['outside_nest_duration']
    out['speeding_event_rate'] = out['speeding_count'] / out['outside_nest_duration']
    out['nest_fraction'] = out['nest_duration'] / (out['nest_duration'] + out['outside_nest_duration'])
    out['nest_mean_duration'] = out['nest_duration'] / out['nest_count']
    out['valid_locomotion_duration_fraction'] = out['valid_locomotion_duration'] / \
        (out['nest_duration'] + out['outside_nest_duration'])
    # social fractions
    out['alone_fraction'] = 1 - (out['weighted_sum'] / out['nest_frames'].replace(0, np.nan))
    out['weighted_co_occupancy'] = out['weighted_sum'] / out['nest_frames'].replace(0, np.nan)
    for act_name in ['feeding','drinking','ramps','s_wall']:
        out[f'{act_name}_together_fraction'] = out[f'{act_name}_together_frames'] / \
            out[f'{act_name}_frames'].replace(0, np.nan)
    return out.reset_index()


if __name__ == '__main__':
    m1 = pd.read_csv('../male_1h_filtered.csv')
    p35 = m1[(m1['age'] == 'P35') & (m1['phase'] == 'active') & (m1['day'] == 1)]

    mdma_matched = build_matched_window_table(p35[p35['time_point'] == 'MDMA'])
    base_matched = build_matched_window_table(p35[p35['time_point'] == 'baseline'])
    print(f"Recovered {len(mdma_matched)} mice for day-1 MDMA-session (matched 7h window)")
    print(f"Baseline day-1 (same matched 7h window) for {len(base_matched)} mice")

    mdma_matched.to_csv('day1_P35_MDMA_matched7h.csv', index=False)
    base_matched.to_csv('day1_P35_baseline_matched7h.csv', index=False)

    check_feats = ['s_wall_duration_fraction','speeding_duration_fraction',
                   'motionless_duration_fraction','total_distance']
    merged = mdma_matched.merge(base_matched[['mouse_ID']+check_feats], on='mouse_ID',
                                 suffixes=('_MDMAday1', '_baseday1'))
    print(merged[['mouse_ID','background','treatment'] +
                 [c for c in merged.columns if any(f in c for f in check_feats)]].head(8).to_string(index=False))
