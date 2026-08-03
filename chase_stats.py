#%%
import os
import pandas as pd
import utils_stats as utils

#%%
CHASE_FOLDER = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\chase"

df_mouse = pd.read_csv(os.path.join(CHASE_FOLDER,'chase_events_by_mouse.csv'))
# cols in events df: mouse, video, batch, time_point, stress, box, date, timestamp, start, end, frames_chasing, frames_chased, event_chasing, event_chased

df_mouse = utils.compute_event_times(df=df_mouse)
# 'recording_start', 'event_start', 'event_end'
df_mouse = utils.add_time_labels(df=df_mouse)
# 'CT_hour', 'phase', 'ZT_day','ZT_hour'
fps = 25
df_mouse['time_chasing'] = df_mouse['frames_chasing'] / fps
df_mouse['time_chased']= df_mouse['frames_chased'] / fps
# cols needed
cols = ['mouse','box','batch','time_point', 'stress',
        'CT_hour', 'phase', 'ZT_day', 'ZT_hour',
        'time_chasing', 'time_chased', 'event_chasing', 'event_chased']

df = df_mouse[cols]

## 1- Group by hour
groupby_cols = ['mouse','box','batch','time_point','stress','ZT_day','phase','ZT_hour','CT_hour']

summary_df = (
    df
    .groupby(groupby_cols)
    .agg(
        total_time_chasing=('time_chasing', 'sum'),
        total_time_chased=('time_chased', 'sum'),
        total_event_chasing=('event_chasing', 'sum'),
        total_event_chased=('event_chased', 'sum')
    )
    .reset_index()
)

## Add conditions and normalize by nest
cond_cols = ['mouse','box','batch','time_point','stress','phase_day','phase','timebin','total_time_outside_s','in_utero_background','mouse_id']
df_cond = pd.read_csv(r"L:\Lopez Laboratory - NEURO\Kelli\Kelli_Nicole_in_utero_psilo_summer_2025\ROIs\all_rois_hour.csv",usecols=cond_cols)
df_cond = df_cond.rename(columns={'phase_day':'ZT_day','timebin':'CT_hour'})
summary_df['ZT_day'] = pd.to_datetime(summary_df['ZT_day']).dt.date
df_cond['ZT_day'] = pd.to_datetime(df_cond['ZT_day']).dt.date

summary_df = summary_df.merge(df_cond,how='outer',on= ['mouse','box','batch','time_point','stress','ZT_day','phase','CT_hour'])

#normalization
summary_df['chasing_norm_total_time_s'] = summary_df['total_time_chasing'] / summary_df['total_time_outside_s']
summary_df['chasing_norm_event_count'] = summary_df['total_event_chasing'] / summary_df['total_time_outside_s']
summary_df['chased_norm_total_time_s'] = summary_df['total_time_chased'] / summary_df['total_time_outside_s']
summary_df['chased_norm_event_count'] = summary_df['total_event_chased'] / summary_df['total_time_outside_s']
summary_df['chasing_avg_time_s'] = summary_df['total_time_chasing'] / summary_df['total_event_chasing']
summary_df['chased_avg_time_s'] = summary_df['total_time_chased'] / summary_df['total_event_chased']
summary_df['chasing_chased_ratio_time'] = summary_df['total_time_chasing'] / (summary_df['total_time_chased']+1)
summary_df['chasing_chased_ratio_event'] = summary_df['total_event_chasing'] / (summary_df['total_event_chased']+1)

summary_df.to_csv(os.path.join(CHASE_FOLDER,'chase_summary_hour.csv'),index=False)

## 2- Group by phase
groupby_cols = ['mouse','box','batch','time_point','stress','ZT_day','phase']

summary_df = (
    df
    .groupby(groupby_cols)
    .agg(
        total_time_chasing=('time_chasing', 'sum'),
        total_time_chased=('time_chased', 'sum'),
        total_event_chasing=('event_chasing', 'sum'),
        total_event_chased=('event_chased', 'sum')
    )
    .reset_index()
)


## Add conditions and normalize by nest
cond_cols = ['mouse','box','batch','time_point','stress','phase_day','phase','total_time_outside_s','in_utero_background','mouse_id']
df_cond = pd.read_csv(r"L:\Lopez Laboratory - NEURO\Kelli\Kelli_Nicole_in_utero_psilo_summer_2025\ROIs\all_rois_phase.csv",usecols=cond_cols)
df_cond = df_cond.rename(columns={'phase_day':'ZT_day'})
summary_df['ZT_day'] = pd.to_datetime(summary_df['ZT_day']).dt.date
df_cond['ZT_day'] = pd.to_datetime(df_cond['ZT_day']).dt.date

set_summary = set(map(tuple, summary_df[groupby_cols].values))
set_cond = set(map(tuple, df_cond[groupby_cols].values))

print("Keys in both:", len(set_summary & set_cond))
print("Only in summary:", len(set_summary - set_cond))
print("Only in cond:", len(set_cond - set_summary))

diff = list(set_summary - set_cond)
print(diff[:10])
diff2 = list(set_cond - set_summary)
print(diff2[:10])

summary_df = summary_df.merge(df_cond,how='outer',on=groupby_cols)

#normalization
summary_df['chasing_norm_total_time_s'] = summary_df['total_time_chasing'] / summary_df['total_time_outside_s']
summary_df['chasing_norm_event_count'] = summary_df['total_event_chasing'] / summary_df['total_time_outside_s']
summary_df['chased_norm_total_time_s'] = summary_df['total_time_chased'] / summary_df['total_time_outside_s']
summary_df['chased_norm_event_count'] = summary_df['total_event_chased'] / summary_df['total_time_outside_s']
summary_df['chasing_avg_time_s'] = summary_df['total_time_chasing'] / summary_df['total_event_chasing']
summary_df['chased_avg_time_s'] = summary_df['total_time_chased'] / summary_df['total_event_chased']
summary_df['chasing_chased_ratio_time'] = summary_df['total_time_chasing'] / (summary_df['total_time_chased']+1)
summary_df['chasing_chased_ratio_event'] = summary_df['total_event_chasing'] / (summary_df['total_event_chased']+1)

summary_df.to_csv(os.path.join(CHASE_FOLDER,'chase_summary_phase.csv'),index=False)

## 3- Group by day
groupby_cols = ['mouse','box','batch','time_point','stress','ZT_day']

summary_df = (
    df
    .groupby(groupby_cols)
    .agg(
        total_time_chasing=('time_chasing', 'sum'),
        total_time_chased=('time_chased', 'sum'),
        total_event_chasing=('event_chasing', 'sum'),
        total_event_chased=('event_chased', 'sum')
    )
    .reset_index()
)


## Add conditions and normalize by nest
cond_cols = ['mouse','box','batch','time_point','stress','phase_day','total_time_outside_s','in_utero_background','mouse_id']
df_cond = pd.read_csv(r"L:\Lopez Laboratory - NEURO\Kelli\Kelli_Nicole_in_utero_psilo_summer_2025\ROIs\all_rois_day.csv",usecols=cond_cols)
df_cond = df_cond.rename(columns={'phase_day':'ZT_day'})
summary_df['ZT_day'] = pd.to_datetime(summary_df['ZT_day']).dt.date
df_cond['ZT_day'] = pd.to_datetime(df_cond['ZT_day']).dt.date

summary_df = summary_df.merge(df_cond,how='outer',on=groupby_cols)

#normalization
summary_df['chasing_norm_total_time_s'] = summary_df['total_time_chasing'] / summary_df['total_time_outside_s']
summary_df['chasing_norm_event_count'] = summary_df['total_event_chasing'] / summary_df['total_time_outside_s']
summary_df['chased_norm_total_time_s'] = summary_df['total_time_chased'] / summary_df['total_time_outside_s']
summary_df['chased_norm_event_count'] = summary_df['total_event_chased'] / summary_df['total_time_outside_s']
summary_df['chasing_avg_time_s'] = summary_df['total_time_chasing'] / summary_df['total_event_chasing']
summary_df['chased_avg_time_s'] = summary_df['total_time_chased'] / summary_df['total_event_chased']
summary_df['chasing_chased_ratio_time'] = summary_df['total_time_chasing'] / (summary_df['total_time_chased']+1)
summary_df['chasing_chased_ratio_event'] = summary_df['total_event_chasing'] / (summary_df['total_event_chased']+1)

summary_df.to_csv(os.path.join(CHASE_FOLDER,'chase_summary_day.csv'),index=False)
