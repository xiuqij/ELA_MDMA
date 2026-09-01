#%%
import os
import pandas as pd
import numpy as np
#import utils_stats as utils
from functools import reduce



#%% REQUIREMENTS
'''
current files:
1. output with the same base cols for merging later: DAY, ZT_hour, phase, box, mouse, time_bin (for sorting), time_window (for plotting)
2. output with desired features for the behavior e.g., total time, event count
3. output in 1,2,3,4,6,12-hr resolutions.
4. all contains nest columns
before merging:
1. add also: time_point, exp, sex, age, mouse_ID, condition, background, treatment before merging
2. make keys
how to merge:
1. take all files in the same resolution
April/male_P35/baseline 
April/male_P35/MDMA
April/female_P42/baseline
April/female_P42/MDMA
July/female_P35/baseline
July/female_P35/MDMA
July/male_P42/baseline
July/male_P42/MDMA

2. define the columns to merge on
3. read only the columns that are needed
example
merged_df = reduce(lambda left,right: pd.merge(left,right,on=merge_keys, how = 'outer'), dfs)
4. HIERARCHY- only on 12h data
5. concat for later use
male
female
male/baseline
male/baseline/active
male/baseline/inactive
female/baseline
female/baseline/active
female/baseline/inactive
male/MDMA
male/MDMA/active
male/MDMA/inactive
female/MDMA
female/MDMA/active
female/MDMA/inactive
'''
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

output_path = "/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA/behavior_dataset"
#%%
# steps 1-3
rois = ['s_wall','ramp1','ramp2','non_wall','woodstick','feeder_prox', 'feeder_dist','water_prox','water_dist']
immobility = ['motionless','speeding']
others = ['chase','locomotion','nest_social','ROI_social']
behaviors = rois + immobility + others
merge_keys = ['day','phase','box','mouse','time_bin','time_window','nest_duration','outside_nest_duration','nest_count','nest_mean_duration']
for b in batches:
    print(b)
    path = os.path.join(main_path,f'{b}_2026')
    for exp in exps[b]:
        print(exp)
        for t in time_points:
            print(t)
            for res in [1,2,3,4,6,12]:
                dfs=[]
                for name in behaviors:
                    if name in rois:
                        folder = os.path.join(path, 'ROIs') 
                    elif name in immobility:
                        folder = os.path.join(path, 'immobility')
                    else:
                        folder = os.path.join(path, name)
                    df = pd.read_csv(os.path.join(folder,exp,t,f'{name}_{res}h.csv'))
                    if (name in rois) or (name in immobility) :
                        df = df.drop(columns=['duration_f'])
                    dfs.append(df)
                merged_df = reduce(lambda left,right: pd.merge(left,right,on=merge_keys, how = 'outer'), dfs)
                merged_df.to_csv(os.path.join(output_path,f'{b}_{exp}_{t}_{res}h.csv'),index=False)


# %%
# step 4 (add hierarchy)
for b in batches:
    print(b)
    for exp in exps[b]:
        print(exp)
        for t in time_points:
            print(t)
            df = pd.read_csv(os.path.join(output_path,f'{b}_{exp}_{t}_12h.csv'))
            df_hierarchy = pd.read_csv(os.path.join(main_path,f'{b}_2026','chase',exp,t,'hierarchy.csv'))
            df = df.merge(df_hierarchy,how='outer',on=['day','phase','box','mouse'])
            df.to_csv(os.path.join(output_path,f'{b}_{exp}_{t}_12h.csv'),index=False)

#%%
# step 5 (concat)
'''
add: time_point, exp, sex, age


- male
- female
male/baseline
male/baseline/active
male/baseline/inactive
female/baseline
female/baseline/active
female/baseline/inactive
male/MDMA
male/MDMA/active
male/MDMA/inactive
female/MDMA
female/MDMA/active
female/MDMA/inactive
'''
#%%
keys = pd.read_csv("/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA/behavior_dataset/keys.csv")
# male
for res in [1,2,3,4,6,12]:
    dfs=[]
    for exp in ['April_male_P35','July_male_P42']:
        exp_name = "_".join(exp.split("_")[1:])
        sex = exp_name.split("_")[0]
        age = exp_name.split("_")[1]
        for t in ['baseline','MDMA']:
            df = pd.read_csv(os.path.join(output_path,'all_exps',f'{exp}_{t}_{res}h.csv'))
            df['time_point'] = t
            df['exp'] = exp_name
            df['sex'] = sex
            df['age'] = age
            df = df.merge(keys,how='left',on=['exp','box','mouse'])
            dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv(os.path.join(output_path,'final','master_feature_table',f'male_{res}h.csv'),index=False)

# female
for res in [1,2,3,4,6,12]:
    dfs=[]
    for exp in ['April_female_P42','July_female_P35']:
        exp_name = "_".join(exp.split("_")[1:])
        sex = exp_name.split("_")[0]
        age = exp_name.split("_")[1]
        for t in ['baseline','MDMA']:
            df = pd.read_csv(os.path.join(output_path,'all_exps',f'{exp}_{t}_{res}h.csv'))
            df['time_point'] = t
            df['exp'] = exp_name
            df['sex'] = sex
            df['age'] = age
            df = df.merge(keys,how='left',on=['exp','box','mouse'])
            dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv(os.path.join(output_path,'final','master_feature_table',f'female_{res}h.csv'),index=False)
#%%
# subset
for res in [1,2,3,4,6,12]:
    for sex in ['male','female']:
        df = pd.read_csv(os.path.join(output_path,'concat',f'{sex}_{res}h.csv'))
        baseline = df[df['time_point']=='baseline']
        baseline_active = baseline[baseline['phase']=='active']
        baseline_inactive = baseline[baseline['phase']=='inactive']
        mdma = df[df['time_point']=='MDMA']
        mdma_active = mdma[mdma['phase']=='active']
        mdma_inactive = mdma[mdma['phase']=='inactive']
        baseline.to_csv(os.path.join(output_path,'concat',f'{sex}_baseline_{res}h.csv'),index=False)
        baseline_active.to_csv(os.path.join(output_path,'concat',f'{sex}_baseline_active_{res}h.csv'),index=False)
        baseline_inactive.to_csv(os.path.join(output_path,'concat',f'{sex}_baseline_inactive_{res}h.csv'),index=False)
        mdma.to_csv(os.path.join(output_path,'concat',f'{sex}_MDMA_{res}h.csv'),index=False)
        mdma_active.to_csv(os.path.join(output_path,'concat',f'{sex}_MDMA_active_{res}h.csv'),index=False)
        mdma_inactive.to_csv(os.path.join(output_path,'concat',f'{sex}_MDMA_inactive_{res}h.csv'),index=False)
# %%
