#%%
'''
Fine-resolution (15min/30min) companion to examine_features/QC_features.py.

Purely additive - does not import from or modify QC_features.py. Reads the
{sex}_15min.csv / {sex}_30min.csv master feature tables produced by
merge_features_fine.py and writes {sex}_15min_flagged.csv / _filtered.csv etc.
alongside the existing resolutions' QC output, on the same NAS mount this session has
access to. Note: the existing QC_features.py itself reads/writes a separate
OneDrive-synced copy of behavior_dataset on the analysis machine, not this NAS mount -
getting these new files there is a manual sync step, same as for the existing
resolutions.
'''
import pandas as pd
import os
import numpy as np
import sys
sys.path.insert(0, "/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA/behavior_classification")
import utils_stats as utils

#%%
master_path = "/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA/behavior_dataset/final/master_feature_table"
QC_output_path = "/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA/behavior_dataset/final/QC_table"

resolutions = [0.25, 0.5]
key_cols = ['day','phase','box','mouse','time_bin','time_window','time_point','exp']

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

def flag_unused_day(df):
    mask = (~df['day'].isin([1,2,3])) | ((df['day']==3) & (df['phase']=='inactive'))
    df['qc_exclude_timebin'] = mask
    return df

def add_box_ID(df):
    is_batch2 = (df['exp']=='male_P42') | (df['exp']=='female_P35')
    df['box_ID'] = df['box'] + (is_batch2 + 1)*10
    return df

#%%
# 1. basic structure (diagnostics only, matches QC_features.py step 1)
for sex in ['male','female']:
    for res in resolutions:
        res_label = utils.format_resolution(res)
        df = pd.read_csv(os.path.join(master_path,f'{sex}_{res_label}.csv'))
        dup_mask = df.duplicated(key_cols,keep=False)
        if dup_mask.sum() == 0:
            print(f'{sex}_{res_label}: no duplicates')
        else:
            print(f'{sex}_{res_label}: {dup_mask.sum()} duplicated entries found.')

#%%
# 2. flag + filter
for sex in ['male','female']:
    for res in resolutions:
        res_label = utils.format_resolution(res)
        df = pd.read_csv(os.path.join(master_path,f'{sex}_{res_label}.csv'))
        df = flag_missing_mouse(df)
        df = flag_locomotion_outlier(df)
        df = flag_unused_day(df)
        df.to_csv(os.path.join(QC_output_path,f'{sex}_{res_label}_flagged.csv'),index=False)
        df_filtered = df[~df['qc_exclude']==1]
        df_filtered = df_filtered[~df_filtered['qc_exclude_timebin']==1]
        df_filtered = add_box_ID(df_filtered)
        df_filtered.to_csv(os.path.join(QC_output_path,f'{sex}_{res_label}_filtered.csv'),index=False)
        print(f'{sex}_{res_label}: wrote flagged ({len(df)} rows) and filtered ({len(df_filtered)} rows)')

# %%
