#%%
import os
import pandas as pd
import numpy as np
import utils_stats as utils

#%%
# Define paths
main_path = "/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA"
path_april = "/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA/April_2026"
path_july = "/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA/July_2026"
exps_april = ['male_P35','female_P42']
exps_july = ['female_P35','male_P42']
batches = ['April','July']
exps = {'April':['male_P35','female_P42'],'July':['female_P35','male_P42']}
time_points = ['baseline','MDMA']

#%%
'''
1. output should have the same base cols for merging later: DAY, ZT_hour, phase, box, mouse, time_bin (for sorting), time_window (for plotting)
2. output should include desired features for the behavior e.g., total time, event count
3. output should be in 1,2,3,4,6,12-hr resolutions.
4. missing data handling - create a common frame and fill 0 (make it on the nest df)
5. add also: time_point, exp, sex, age, mouse_ID, condition, background, treatment before merging
'''
#%%
# nest
complete_cols = ['video','date','ZT_day','day','ZT_hour','CT_hour','phase','box','mouse','event_start','event_end','start_frame','end_frame','duration_f']
keep_cols = ['day','ZT_hour','phase','box','mouse','duration_f']
base_cols = ['day','phase','box','mouse']
for b in batches:
    print(b)
    path = os.path.join(main_path,f'{b}_2026')
    for exp in exps[b]:
        print(exp)
        for t in time_points:
            print(t)
            df_4h = pd.read_csv(os.path.join(path,'nest',exp,t,'nest_events.csv'))
            utils.convert_to_1h(df_4h,keep_cols=complete_cols).to_csv(os.path.join(path,'nest',exp,t,'nest_1h_sanity_check.csv'),index=False)
            df_1h = utils.convert_to_1h(df_4h,keep_cols=keep_cols)
            for res in [1,2,3,4,6,12]:
                df_res = utils.regroup_by_timebin(df_1h, resolution = res, group_base = base_cols, nest = True)
                df_res.to_csv(os.path.join(path,'nest',exp,t,f'nest_{res}h.csv'),index=False)
'''
# nest correction (April, MDMA, male - recording started late)
for res in [1,2,3,4,6,12]:
    csv_path = f'/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA/April_2026/nest/male_P35/MDMA/nest_{res}h.csv'
    df = pd.read_csv(csv_path)
    df['day'] = df['day'] + 1
    df.to_csv(csv_path,index=False)
'''

#%%
# ROIs
complete_cols = ['video','date','ZT_day','day','ZT_hour','CT_hour','phase','box','mouse','event_start','event_end','start_frame','end_frame','duration_f']
keep_cols = ['day','ZT_hour','phase','box','mouse','duration_f']
base_cols = ['day','phase','box','mouse']
rois = ['s_wall','ramp1','ramp2','non_wall','woodstick','feeder_prox', 'feeder_dist','water_prox','water_dist']
for b in batches:
    print(b)
    path = os.path.join(main_path,f'{b}_2026')
    for exp in exps[b]:
        print(exp)
        for t in time_points:
            print(t)
            for roi in rois:
                df_4h = pd.read_csv(os.path.join(path,'ROIs',exp,t,f'{roi}_events.csv'))
                
                utils.convert_to_1h(df_4h,keep_cols=complete_cols).to_csv(os.path.join(path,'ROIs',exp,t,f'{roi}_1h_sanity_check.csv'),index=False)
                df_1h = utils.convert_to_1h(df_4h,keep_cols=keep_cols)
                if b=='April' and exp=='male_P35' and t=='MDMA':
                    df_1h['day'] = df_1h['day'] + 1
                for res in [1,2,3,4,6,12]:
                    df_res = utils.regroup_by_timebin(df_1h, resolution = res, group_base = base_cols, nest = False)
                    df_res = utils.normalize_by_nest(df_res,
                                                     nest_df=pd.read_csv(os.path.join(path,'nest',exp,t,f'nest_{res}h.csv')))
                    df_res = df_res.rename(columns = {'duration':f'{roi}_duration',
                                                      'count':f'{roi}_count',
                                                      'mean_duration':f'{roi}_mean_duration',
                                                      'duration_fraction':f'{roi}_duration_fraction',
                                                      'event_rate':f'{roi}_event_rate'})
                    df_res.to_csv(os.path.join(path,'ROIs',exp,t,f'{roi}_{res}h.csv'),index=False)

#%%
# chases
# run chase_pair_to_mouse.py and chase_stast.py
#%%
# hierarchies
# already computed - 12h is the smallest unit
#%%
# locomotion
# run locomotion_stats.py
#%%
# motionless
complete_cols = ['video','date','ZT_day','day','ZT_hour','CT_hour','phase','box','mouse','event_start','event_end','start_frame','end_frame','duration_f']
keep_cols = ['day','ZT_hour','phase','box','mouse','duration_f']
base_cols = ['day','phase','box','mouse']
rois = ['s_wall','ramp1','ramp2','non_wall','woodstick','feeder_prox', 'feeder_dist','water_prox','water_dist']
for b in batches:
    print(b)
    path = os.path.join(main_path,f'{b}_2026')
    for exp in exps[b]:
        print(exp)
        for t in time_points:
            print(t)
            df_4h = pd.read_csv(os.path.join(path, 'immobility',exp,t,'motionless_events.csv'))
            df_4h["date"] = df_4h["video"].str[:10]
            df_4h["timestamp"] = df_4h["video"].str[11:19]
            utils.convert_to_1h(df_4h,keep_cols=complete_cols).to_csv(os.path.join(path, 'immobility',exp,t,'motionless_1h_sanity_check.csv'),index=False)
            df_1h = utils.convert_to_1h(df_4h,keep_cols=keep_cols)
            if b=='April' and exp=='male_P35' and t=='MDMA':
                df_1h['day'] = df_1h['day'] + 1
            for res in [1,2,3,4,6,12]:
                df_res = utils.regroup_by_timebin(df_1h, resolution = res, group_base = base_cols, nest = False)
                df_res = utils.normalize_by_nest(df_res,
                                                nest_df=pd.read_csv(os.path.join(path,'nest',exp,t,f'nest_{res}h.csv')))
                df_res = df_res.rename(columns = {'duration':'motionless_duration','count':'motionless_count',
                                                'mean_duration':'motionless_mean_duration',
                                                'duration_fraction':'motionless_duration_fraction',
                                                'event_rate':'motionless_event_rate'})
                df_res.to_csv(os.path.join(path,'immobility',exp,t,f'motionless_{res}h.csv'),index=False)


#%%
# speeding
complete_cols = ['video','date','ZT_day','day','ZT_hour','CT_hour','phase','box','mouse','event_start','event_end','start_frame','end_frame','duration_f']
keep_cols = ['day','ZT_hour','phase','box','mouse','duration_f']
base_cols = ['day','phase','box','mouse']
rois = ['s_wall','ramp1','ramp2','non_wall','woodstick','feeder_prox', 'feeder_dist','water_prox','water_dist']
for b in batches:
    print(b)
    path = os.path.join(main_path,f'{b}_2026')
    for exp in exps[b]:
        print(exp)
        for t in time_points:
            print(t)
            df_4h = pd.read_csv(os.path.join(path, 'immobility',exp,t,'speeding_events.csv'))
            df_4h["date"] = df_4h["video"].str[:10]
            df_4h["timestamp"] = df_4h["video"].str[11:19]
            utils.convert_to_1h(df_4h,keep_cols=complete_cols).to_csv(os.path.join(path, 'immobility',exp,t,'speeding_1h_sanity_check.csv'),index=False)
            df_1h = utils.convert_to_1h(df_4h,keep_cols=keep_cols)
            if b=='April' and exp=='male_P35' and t=='MDMA':
                df_1h['day'] = df_1h['day'] + 1
            for res in [1,2,3,4,6,12]:
                df_res = utils.regroup_by_timebin(df_1h, resolution = res, group_base = base_cols, nest = False)
                df_res = utils.normalize_by_nest(df_res,
                                                nest_df=pd.read_csv(os.path.join(path,'nest',exp,t,f'nest_{res}h.csv')))
                df_res = df_res.rename(columns = {'duration':'speeding_duration','count':'speeding_count',
                                                'mean_duration':'speeding_mean_duration',
                                                'duration_fraction':'speeding_duration_fraction',
                                                'event_rate':'speeding_event_rate'})
                df_res.to_csv(os.path.join(path,'immobility',exp,t,f'speeding_{res}h.csv'),index=False)


#%%
# social_nest and social_ROI (already computed but fixing the columns for merge)
# merge_keys = ['day','phase','box','mouse','time_bin','time_window','nest_duration','outside_nest_duration','nest_count','nest_mean_duration']
def add_labels(df):
    df["ZT_day"] = pd.to_datetime(df["ZT_day"])
    first_day = df["ZT_day"].min()
    df["day"] = ((df["ZT_day"] - first_day).dt.days
    + 1
    - (df["time_bin"] <= 12).astype(int)
    )
    df["phase"] = (df["time_bin"] > 12).map( {True: "active", False: "inactive"} ) 

    return df

def merge_nest(df,nest_df,group_cols=['day','phase','box','mouse','time_bin','time_window']):
    # duration, count, mean_duation, outside_nest_duration, duration_fraction, event_rate
    nest_df = nest_df.rename(columns = {'total_time':'nest_duration','count':'nest_count','avg_time':'nest_mean_duration','outside_total_time':'outside_nest_duration'})
    #nest_df = nest_df.rename(columns = {'duration':'nest_duration','count':'nest_count','mean_duration':'nest_mean_duration'})
    nest_cols = group_cols + ['nest_duration','outside_nest_duration','nest_count','nest_mean_duration']
    df = df.merge(nest_df[nest_cols],how='outer',on=group_cols)
    
    return df

for b in batches:
    print(b)
    path = os.path.join(main_path,f'{b}_2026')
    for exp in exps[b]:
        print(exp)
        for t in time_points:
            print(t)
            for res in [1,2,3,4,6,12]:
                df_nest = pd.read_csv(os.path.join(path,'nest',exp,t,f'nest_{res}h.csv'))
                for name in ['nest_social','ROI_social']:
                    df = pd.read_csv(os.path.join(path,name,exp,t,f'{name}_{res}h.csv'))
                    df = add_labels(df)
                    df.to_csv(os.path.join(path,name,exp,t,f'{name}_{res}h_check.csv'),index=False)
                    df = df.drop(columns=['ZT_day'])
                    df = merge_nest(df,nest_df=df_nest)
                    df.to_csv(os.path.join(path,name,exp,t,f'{name}_{res}h.csv'),index=False)

# %%
