#%%
'''
Fine-resolution (15min/30min) companion to merge_features.py steps 1-3 and 5.

Purely additive - does not import from or modify merge_features.py. Reads the
{name}_15min.csv / {name}_30min.csv files produced by regroup_timebins_fine.py, and
writes new merged files alongside the existing ones, at the same locations
merge_features.py actually uses (behavior_dataset/all_exps/, behavior_dataset/final/
master_feature_table/) - note the current merge_features.py's step 1-3 writes to
`output_path` (behavior_dataset/ root) directly, but the files that step 5 actually
reads live in behavior_dataset/all_exps/, so that's the real, working location this
mirrors.

Step 4 (hierarchy merge, 12h-only) is not replicated: normDS/rank only exist at 12h
resolution - expected/documented in prep.py and temporal_dynamics.py, not a gap here.

The "subset" step at the end of merge_features.py (writing to behavior_dataset/concat/)
is also not replicated: that folder is currently empty even for the existing 1-12h
resolutions, and prep.filter_data()/temporal_dynamics.load_temporal_data() already do
all baseline/MDMA/phase subsetting at load time - so it appears unused today.
'''
import os
import pandas as pd
from functools import reduce
import sys
sys.path.insert(0, "/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA/behavior_classification")
import utils_stats as utils

#%%
main_path = "/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA"
batches = ['April','July']
exps = {'April':['male_P35','female_P42'],'July':['female_P35','male_P42']}
time_points = ['baseline','MDMA']
resolutions = [0.25, 0.5]

output_path = "/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA/behavior_dataset"
all_exps_path = os.path.join(output_path, 'all_exps')

rois = ['s_wall','ramp1','ramp2','non_wall','woodstick','feeder_prox', 'feeder_dist','water_prox','water_dist']
immobility = ['motionless','speeding']
others = ['chase','locomotion','nest_social','ROI_social']
behaviors = rois + immobility + others
merge_keys = ['day','phase','box','mouse','time_bin','time_window','nest_duration','outside_nest_duration','nest_count','nest_mean_duration']

#%%
# steps 1-3: per (batch, exp, time_point, resolution) merge
for b in batches:
    print(b)
    path = os.path.join(main_path,f'{b}_2026')
    for exp in exps[b]:
        print(exp)
        for t in time_points:
            print(t)
            for res in resolutions:
                res_label = utils.format_resolution(res)
                dfs=[]
                for name in behaviors:
                    if name in rois:
                        folder = os.path.join(path, 'ROIs')
                    elif name in immobility:
                        folder = os.path.join(path, 'immobility')
                    else:
                        folder = os.path.join(path, name)
                    df = pd.read_csv(os.path.join(folder,exp,t,f'{name}_{res_label}.csv'))
                    if (name in rois) or (name in immobility):
                        df = df.drop(columns=['duration_f'])
                    dfs.append(df)
                merged_df = reduce(lambda left,right: pd.merge(left,right,on=merge_keys, how = 'outer'), dfs)
                merged_df.to_csv(os.path.join(all_exps_path,f'{b}_{exp}_{t}_{res_label}.csv'),index=False)

# %%
# step 5: concat across exps into male_{res_label}.csv / female_{res_label}.csv
keys = pd.read_csv(os.path.join(output_path,"keys.csv"))

sex_exp_groups = {
    'male': ['April_male_P35','July_male_P42'],
    'female': ['April_female_P42','July_female_P35'],
}

for res in resolutions:
    res_label = utils.format_resolution(res)
    for sex, exp_list in sex_exp_groups.items():
        dfs=[]
        for exp in exp_list:
            exp_name = "_".join(exp.split("_")[1:])
            s = exp_name.split("_")[0]
            age = exp_name.split("_")[1]
            for t in time_points:
                df = pd.read_csv(os.path.join(all_exps_path,f'{exp}_{t}_{res_label}.csv'))
                df['time_point'] = t
                df['exp'] = exp_name
                df['sex'] = s
                df['age'] = age
                df = df.merge(keys,how='left',on=['exp','box','mouse'])
                dfs.append(df)
        df = pd.concat(dfs)
        df.to_csv(os.path.join(output_path,'final','master_feature_table',f'{sex}_{res_label}.csv'),index=False)

# %%
