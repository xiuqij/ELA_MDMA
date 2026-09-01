#%%
import os
import pandas as pd
import numpy as np
import utils_stats as utils

#%%
'''
cols in output: video,box,mouse,date,timestamp,distance,speed,time_in_seconds,angular_velocity,acceleration
features to include:
- total_distance (sum of distance)
- valid_locomotion_duration (duration represented by valid distance/speed observations)
- mean_speed (mean of speed)
- median_speed (median of speed)
- mean_abs_angular_velocity (mean of abs(angular_velocity))
- mean_abs_accleration
'''

#%%
def locomotion_regroup_by_timebin(df_1h, resolution, group_base = ['day','phase','box','mouse']):
    df = df_1h.copy()
    df = utils.add_timebin_labels(df,resolution)
    group_cols = group_base + ['time_bin','time_window']
    #locomotion data are sampled once per second, 
    df['valid_locomotion'] = df['speed'].notna()
    df['abs_angular_velocity'] = df['angular_velocity'].abs()
    df['abs_acceleration'] = df['acceleration'].abs()

    group_df = (
        df.groupby(group_cols, observed=True)
        .agg(
            total_distance=('distance', 'sum'),

            valid_locomotion_duration=('valid_locomotion', 'sum'),

            mean_speed=('speed', 'mean'),
            median_speed=('speed', 'median'),

            mean_abs_angular_velocity=(
                'abs_angular_velocity',
                'mean'
            ),

            mean_abs_acceleration=(
                'abs_acceleration',
                'mean'
            )
        )
        .reset_index()
    )

    keep_cols = ['day','phase','box','mouse','time_bin','time_window','total_distance','valid_locomotion_duration','mean_speed','median_speed','mean_abs_angular_velocity','mean_abs_acceleration']

    return group_df[keep_cols]


def locomotion_normalize_by_nest(df,nest_df,group_cols=['day','phase','box','mouse','time_bin','time_window']):
    # duration, count, mean_duation, outside_nest_duration, duration_fraction, event_rate
    nest_df = nest_df.rename(columns = {'total_time':'nest_duration','count':'nest_count','avg_time':'nest_mean_duration','outside_total_time':'outside_nest_duration'})
    #nest_df = nest_df.rename(columns = {'duration':'nest_duration','count':'nest_count','mean_duration':'nest_mean_duration'})
    nest_cols = group_cols + ['nest_duration','outside_nest_duration','nest_count','nest_mean_duration']
    df = df.merge(nest_df[nest_cols],how='outer',on=group_cols)
    df['valid_locomotion_duration_fraction'] = df['valid_locomotion_duration'] / df['outside_nest_duration']
    
    return df

def compute_event_time_locomotion(df):
    # Recording/video start time
    df["recording_start"] = pd.to_datetime(
        df["date"].astype(str) + " " +
        df["timestamp"].astype(str).str.replace("-", ":")
    )

    # Actual timestamp of each locomotion observation
    df["event_start"] = (
        df["recording_start"] +
        pd.to_timedelta(df["time_in_seconds"], unit="s")
    )

    return df
#%%
# add labels -> regroup
main_path = "/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA"
batches = ['April','July']
exps = {'April':['male_P35','female_P42'],'July':['female_P35','male_P42']}
time_points = ['baseline','MDMA']

cols_complete = ['video','date','ZT_day','day','ZT_hour','CT_hour','phase','box','mouse','event_start','distance','speed','time_in_seconds','angular_velocity','acceleration']
cols_1h = ['day','ZT_hour','phase','box','mouse','distance','speed','angular_velocity','acceleration']
for b in batches:
    print(b)
    path = os.path.join(main_path,f'{b}_2026','locomotion')
    for exp in exps[b]:
        print(exp)
        for t in time_points:
            print(t)
            df = pd.read_csv(os.path.join(path,exp,t,'locomotion.csv'))
            df = compute_event_time_locomotion(df)
            # skipped the 1h separation since chasing events are usually short
            df = utils.add_time_labels(df)
            df = utils.add_day_order(df)
            if b=='April' and exp=='male_P35' and t=='MDMA':
                df['day'] = df['day'] + 1
            df[cols_complete].to_csv(os.path.join(path,exp,t,'locomotion_1h_sanity_check.csv'),index=False)
            df_1h = df[cols_1h]
            for res in [1,2,3,4,6,12]:
                df_res = locomotion_regroup_by_timebin(df_1h,resolution=res)
                df_res = locomotion_normalize_by_nest(df_res,
                                                 nest_df=pd.read_csv(os.path.join(main_path,f'{b}_2026','nest',exp,t,f'nest_{res}h.csv')))
                df_res.to_csv(os.path.join(path,exp,t,f'locomotion_{res}h.csv'),index=False)

# %%
