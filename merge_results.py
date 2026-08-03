#%%
import os
import pandas as pd
import utils_stats as utils
from functools import reduce


#%%
exps = ['male_P35','female_P42']   #LIST OF FOLDERS
timepoints = ['baseline','MDMA_acute','MDMA']
rois = ['s_wall','ramp1','ramp2','non_wall','woodstick','feeder_prox', 'feeder_dist','water_prox','water_dist']


rename_cols = {'mouse':'mouse_color','box':'SB'}

#%% stats for nest
nest_folder = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\nest"
for exp in exps:
    for tp in timepoints:
        nest_events = pd.read_csv(os.path.join(nest_folder,exp,tp,'nest_events.csv'))
        nest_events_1h = utils.convert_to_1h(df_4h=nest_events)
        nest_events_1h = nest_events_1h.rename(columns=rename_cols)
        nest_events_1h.to_csv(os.path.join(nest_folder,exp,tp,'nest_events_1h.csv'),index=False)
        for unit in ['hour','phase','day']:
            stats_df = utils.get_summary_stats(raw_df=nest_events_1h,group_by=unit,nest=True,base_cols=['mouse_color','SB'])
            stats_df.to_csv(os.path.join(nest_folder,exp,tp,f'nest_stats_{unit}.csv'),index=False)


#%% stats for rois
# event cols: video,mouse,start_frame,end_frame,duration,box,date,timestamp
roi_folder = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\ROIs"
for exp in exps:
    for tp in timepoints:
        for roi in rois:
            roi_events = pd.read_csv(os.path.join(roi_folder,exp,tp,f'{roi}_events.csv'))
            roi_events_1h = utils.convert_to_1h(df_4h=roi_events)
            roi_events_1h = roi_events_1h.rename(columns=rename_cols)
            roi_events_1h.to_csv(os.path.join(roi_folder,exp,tp,f'{roi}_events_1h.csv'),index=False)
            for unit in ['hour','phase','day']:
                stats_df = utils.get_summary_stats(raw_df=roi_events_1h,group_by=unit,nest=False,
                                                   nest_df=pd.read_csv(os.path.join(nest_folder,exp,tp,f'nest_stats_{unit}.csv')),
                                                   base_cols=['mouse_color','SB'])
                stats_df.to_csv(os.path.join(roi_folder,exp,tp,f'{roi}_stats_{unit}.csv'),index=False)

# %% merge results
# cols :mouse_color,SB,ZT_day,phase,ZT_hour,CT_hour,event_count,total_time_s,avg_time_s,total_time_outside_s,nest_entries,nest_avg_time_s,norm_total_time_s,norm_event_count

save_path = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026"
time_cols = {'hour':['ZT_day','phase','ZT_hour','CT_hour'],
             'phase':['ZT_day','phase'],
             'day':['ZT_day']}
for unit in ['hour','phase','day']:
    cols = time_cols[unit]
    merge_keys = ['mouse_color','SB','time_point','total_time_outside_s','nest_entries','nest_avg_time_s'] + cols
    for exp in exps:
        merged_dfs = []
        for tp in timepoints:
            dfs = []
            for roi in rois:
                df = pd.read_csv(os.path.join(roi_folder,exp,tp,f'{roi}_stats_{unit}.csv'))
                df['time_point'] = tp 
                df = df.rename(columns={'event_count':f'{roi}_event_count',
                                        'total_time_s':f'{roi}_total_time_s',
                                        'avg_time_s':f'{roi}_avg_time_s',
                                        'norm_total_time_s':f'{roi}_norm_total_time_s',
                                        'norm_event_count':f'{roi}_norm_event_count'})
                
                dfs.append(df)
            merged_df = reduce(lambda left,right: pd.merge(left,right,
                                                       on=merge_keys, how = 'outer'), dfs)
            merged_dfs.append(merged_df)
        final_df = pd.concat(merged_dfs)
        final_df.to_csv(os.path.join(save_path,f'all_rois_{exp}_{unit}.csv'),index=False)
# %% Add conditions
key_f42 = pd.read_csv(r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\key_f42.csv")
key_m35 = pd.read_csv(r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\keys_m35.csv")
for unit in ['hour','phase','day']:
    male = pd.read_csv(os.path.join(save_path,f'all_rois_male_P35_{unit}.csv'))
    male = male.merge(key_m35,how='left',on=['mouse_color','SB'])
    male.to_csv(os.path.join(save_path,f'rois_male_P35_{unit}.csv'),index=False)
    female = pd.read_csv(os.path.join(save_path,f'all_rois_female_P42_{unit}.csv'))
    female = female.merge(key_f42,how='left',on=['mouse_color','SB'])
    female.to_csv(os.path.join(save_path,f'rois_female_P42_{unit}.csv'),index=False)
# %%
