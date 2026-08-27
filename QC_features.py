#%%
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%
master_path = "/Users/xiuqi.ji/Library/CloudStorage/OneDrive-KarolinskaInstitutet/MDMA_ELA/social_box_2026/github_repo/behavior_dataset/final/master_feature_table"
QC_output_path = "/Users/xiuqi.ji/Library/CloudStorage/OneDrive-KarolinskaInstitutet/MDMA_ELA/social_box_2026/github_repo/behavior_dataset/final/QC_table"
#%%
resolutions = [1,2,3,4,6,12]
info_cols = ['day', 'phase', 'box', 'mouse', 'time_bin', 'time_window','time_point', 'exp', 'sex', 'age', 'mouse_ID', 'background', 'treatment', 'condition']

nest_cols = ['nest_duration', 'outside_nest_duration', 'nest_count', 'nest_mean_duration']
roi_cols = ['s_wall_count', 's_wall_duration', 's_wall_mean_duration', 's_wall_duration_fraction', 's_wall_event_rate', 
'ramp1_count', 'ramp1_duration', 'ramp1_mean_duration', 'ramp1_duration_fraction', 'ramp1_event_rate', 
'ramp2_count', 'ramp2_duration', 'ramp2_mean_duration', 'ramp2_duration_fraction', 'ramp2_event_rate', 
'non_wall_count', 'non_wall_duration', 'non_wall_mean_duration', 'non_wall_duration_fraction', 'non_wall_event_rate', 
'woodstick_count', 'woodstick_duration', 'woodstick_mean_duration', 'woodstick_duration_fraction', 'woodstick_event_rate', 
'feeder_prox_count', 'feeder_prox_duration', 'feeder_prox_mean_duration', 'feeder_prox_duration_fraction', 'feeder_prox_event_rate', 
'feeder_dist_count', 'feeder_dist_duration', 'feeder_dist_mean_duration', 'feeder_dist_duration_fraction', 'feeder_dist_event_rate', 
'water_prox_count', 'water_prox_duration', 'water_prox_mean_duration', 'water_prox_duration_fraction', 'water_prox_event_rate', 
'water_dist_count', 'water_dist_duration', 'water_dist_mean_duration', 'water_dist_duration_fraction', 'water_dist_event_rate']
chase_cols = ['chasing_duration', 'chasing_count', 'chasing_mean_duration', 'chased_duration', 'chased_count', 'chased_mean_duration', 
'chasing_duration_fraction', 'chasing_event_rate', 'chased_duration_fraction', 'chased_event_rate',
'chasing_duration_ratio', 'chased_duration_ratio', 'chasing_event_ratio', 'chased_event_ratio']
hierarchy_cols = ['normDS', 'rank']
locomotion_cols = ['total_distance', 'valid_locomotion_duration', 'mean_speed', 'median_speed', 'mean_abs_angular_velocity', 'mean_abs_acceleration', 'valid_locomotion_duration_fraction']
motionless_cols = ['motionless_count', 'motionless_duration', 'motionless_mean_duration', 'motionless_duration_fraction', 'motionless_event_rate']
speeding_cols = ['speeding_count', 'speeding_duration', 'speeding_mean_duration', 'speeding_duration_fraction', 'speeding_event_rate']
social_cols = ['nest_frames', 'weighted_sum', 'alone_sum', 'weighted_co_occupancy', 'alone_fraction', 'feeding_frames', 'drinking_frames', 'ramps_frames', 's_wall_frames', 'feeding_together_frames', 'drinking_together_frames', 'ramps_together_frames', 's_wall_together_frames', 'feeding_alone_frames', 'drinking_alone_frames', 'ramps_alone_frames', 's_wall_alone_frames', 'feeding_together_fraction', 'feeding_alone_fraction', 'drinking_together_fraction', 'drinking_alone_fraction', 'ramps_together_fraction', 'ramps_alone_fraction', 's_wall_together_fraction', 's_wall_alone_fraction']

#%%
# 1. Basic structure

key_cols = ['day','phase','box','mouse','time_bin','time_window','time_point','exp']

for sex in ['male','female']:
    for res in resolutions:
        df = pd.read_csv(os.path.join(master_path,f'{sex}_{res}h.csv'))
        print("\nChecking for duplication")
        dup_mask = df.duplicated(key_cols,keep=False)
        if dup_mask.sum() == 0:
            print(f'{sex}_{res}h: passed')  
        else: 
            print(f'{sex}_{res}h: duplicated entries found.')
            print(dup_mask.sum())
            df['qc_duplicate'] = dup_mask
            print(df.loc[df['qc_duplicate'],key_cols])
        print("check mouse composition")
        composition = (
            df[['exp', 'box', 'mouse_ID', 'background']]
            .drop_duplicates()
            .groupby(['exp', 'box'])
            .agg(
                n_mice=('mouse_ID', 'nunique'),
                n_background=('background', 'nunique')
            )
        )
        if composition[composition['n_mice'] !=4].empty:
            print(f'{sex}_{res}h: passed')
        else:
            print(f'{sex}_{res}h: not passed')
        print("check treament assignment")
        treatment_by_box = (
            df[['exp', 'box', 'treatment']]
            .drop_duplicates()
            .groupby(['exp', 'treatment'])
            ['box']
            .nunique()
        )
        print(treatment_by_box)



# %%
# 2. completeness
group_cols = ['exp','time_point','day','phase','box','mouse']

for sex in ['male','female']:
    for res in resolutions:
        df = pd.read_csv(os.path.join(master_path,f'{sex}_{res}h.csv'))
        counts = (
            df.groupby(
                group_cols
            )['time_bin']
            .nunique()
        )
        print(f'\n{sex}_{res}h:')
        print(counts.rename('n_bins').reset_index().to_string(index=False))
# %%
# flag impossible values
# male_P42, MDMA, box 8, blue
# extreme locomotion values
# flag timebin that wont be used for now (day 0 and day 3 inactive)
def flag_locomotion_outlier(df):
    Q1 = df['mean_speed'].quantile(.25)
    Q3 = df['mean_speed'].quantile(.75)
    IQR = Q3 - Q1
    upper = Q3 + 3*IQR
    df['qc_speed_outlier'] = df['mean_speed'] > upper
    return df
def flag_missing_mouse(df):
    mask = (df['exp']=='male_P42') & (df['time_point']=='MDMA') & (df['box']==8) & (df['mouse'] == 'blue')
    df['qc_exclude'] = mask
    return df
def flag_statictical_outlier(df,col):
    median = df[col].median
    mad = np.median(np.abs(df[col] - median))
    if mad ==0:
        df[f'qc_{col}_statistical_outlier'] = False
        return df
    robust_z = 0.6745 * (df[col] - median)/mad
    df[f'qc_{col}_statistical_outlier'] = robust_z.abs() > 3.5
    return df

def flag_unused_day(df):
    mask = (~df['day'].isin([1,2,3])) | ((df['day']==3) & (df['phase']=='inactive'))
    df['qc_exclude_timebin'] = mask
    return df

def add_box_ID(df):
    # male_P35, female_P42 -> 1 ; 
    is_batch2 = (df['exp']=='male_P42') | (df['exp']=='female_P35')
    df['box_ID'] = df['box'] + (is_batch2 + 1)*10
    return df
#%%
for sex in ['male','female']:
    for res in resolutions:
        df = pd.read_csv(os.path.join(master_path,f'{sex}_{res}h.csv'))
        df = flag_missing_mouse(df)
        df = flag_locomotion_outlier(df)
        df = flag_unused_day(df)
        df.to_csv(os.path.join(QC_output_path,f'{sex}_{res}h_flagged.csv'),index=False)
        df_filtered = df[~df['qc_exclude']==1]
        df_filtered = df_filtered[~df_filtered['qc_exclude_timebin']==1]
        df_filtered = add_box_ID(df_filtered)
        df_filtered.to_csv(os.path.join(QC_output_path,f'{sex}_{res}h_filtered.csv'),index=False)
# %%
