#%%
import os
import pandas as pd
import numpy as np
import utils_stats as utils

#%%
# customized functions
def chase_regroup_by_timebin(df_1h,resolution,fps=25,group_base = ['day','phase','box','mouse']):
    '''regroup and compute sums and event counts. 
    '''
    df = df_1h.copy()
    df = utils.add_timebin_labels(df,resolution)
    group_cols = group_base + ['time_bin','time_window']


    group_df = (
        df.groupby(group_cols,observed=True)
        .agg(
            frames_chasing = ('frames_chasing','sum'),
            frames_chased = ('frames_chased','sum'),
            chasing_count = ('count_chasing','sum'),
            chased_count = ('count_chased','sum')
        )
        .reset_index()
    )
    group_df['chasing_duration'] = group_df['frames_chasing'] / fps
    group_df['chased_duration'] = group_df['frames_chased'] / fps
    group_df['chasing_mean_duration'] = np.where(group_df['chasing_count'] > 0, group_df['chasing_duration'] / group_df['chasing_count'],0)
    group_df['chased_mean_duration'] = np.where(group_df['chased_count'] > 0, group_df['chased_duration'] / group_df['chased_count'],0)

    keep_cols = ['day','phase','box','mouse','time_bin','time_window','chasing_duration','chasing_count','chasing_mean_duration','chased_duration','chased_count','chased_mean_duration']
    return group_df[keep_cols]

def chase_normalize_by_nest(df,nest_df,group_cols=['day','phase','box','mouse','time_bin','time_window']):
    # duration, count, mean_duation, outside_nest_duration, duration_fraction, event_rate
    nest_df = nest_df.rename(columns = {'total_time':'nest_duration','count':'nest_count','avg_time':'nest_mean_duration','outside_total_time':'outside_nest_duration'})
    #nest_df = nest_df.rename(columns = {'duration':'nest_duration','count':'nest_count','mean_duration':'nest_mean_duration'})
    nest_cols = group_cols + ['nest_duration','outside_nest_duration','nest_count','nest_mean_duration']
    df = df.merge(nest_df[nest_cols],how='outer',on=group_cols)
    df['chasing_duration_fraction'] = df['chasing_duration'] / df['outside_nest_duration']
    df['chasing_event_rate'] = df['chasing_count'] / df['outside_nest_duration']
    df['chased_duration_fraction'] = df['chased_duration'] / df['outside_nest_duration']
    df['chased_event_rate'] = df['chased_count'] / df['outside_nest_duration']

    df['chasing_duration_ratio'] = df['chasing_duration'] / (df['chasing_duration']+df['chased_duration'])
    df['chased_duration_ratio'] = df['chased_duration'] / (df['chasing_duration']+df['chased_duration'])
    df['chasing_event_ratio'] = df['chasing_count'] / (df['chasing_count']+df['chased_count'])
    df['chased_event_ratio'] = df['chased_count'] / (df['chasing_count']+df['chased_count'])

    return df

#%%
# add labels -> regroup
main_path = "/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA"
batches = ['April','July']
exps = {'April':['male_P35','female_P42'],'July':['female_P35','male_P42']}
time_points = ['baseline','MDMA']

cols_complete = ['video','date','ZT_day','day','ZT_hour','CT_hour','phase','box','mouse','event_start','event_end','start_frame','end_frame','frames_chasing','frames_chased','count_chasing','count_chased']
cols_1h = ['day','ZT_hour','phase','box','mouse','frames_chasing','frames_chased','count_chasing','count_chased']
for b in batches:
    print(b)
    path_chase = os.path.join(main_path,f'{b}_2026','chase')
    for exp in exps[b]:
        print(exp)
        for t in time_points:
            print(t)
            df = pd.read_csv(os.path.join(path_chase,exp,t,'chase_events_by_mouse.csv'))
            df = utils.compute_event_times(df)
            # skipped the 1h separation since chasing events are usually short
            df = utils.add_time_labels(df)
            df = utils.add_day_order(df)
            if b=='April' and exp=='male_P35' and t=='MDMA':
                df['day'] = df['day'] + 1
            df[cols_complete].to_csv(os.path.join(path_chase,exp,t,'chase_1h_sanity_check.csv'),index=False)
            df_1h = df[cols_1h]
            for res in [1,2,3,4,6,12]:
                df_res = chase_regroup_by_timebin(df_1h,resolution=res)
                df_res = chase_normalize_by_nest(df_res,
                                                 nest_df=pd.read_csv(os.path.join(main_path,f'{b}_2026','nest',exp,t,f'nest_{res}h.csv')))
                df_res.to_csv(os.path.join(path_chase,exp,t,f'chase_{res}h.csv'),index=False)

# %%
