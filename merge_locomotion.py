#%%
import pandas as pd
import os

#%%
# paths
locomotion_path = '/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA/April_2026/locomotion'
exps = ['male_P35','female_P42']
time_points = ['baseline','MDMA']

#%%
for exp in exps:
    for t in time_points:
        pickle_path = os.path.join(locomotion_path, exp, t, 'locomotion.pickle')
        raw_dict = pd.read_pickle(pickle_path)
        locomotion_dfs = []
        for k,v in raw_dict.items():
            # cols to have: video, mouse, sb, timepoint, 
            locomotion_df = v['locomotion']
            if locomotion_df.empty:
                locomotion_df = pd.DataFrame(columns=['video','box','mouse','date','timestamp','distance', 'speed', 'time_in_seconds', 'angular_velocity','acceleration'])
            locomotion_df['video'] = k
            locomotion_df['box'] = k.split("_")[-1]
            locomotion_df['date'] = k.split("_")[0]
            locomotion_df['timestamp'] = k.split("_")[1]
            locomotion_df= locomotion_df[['video','box','mouse','date','timestamp','distance', 'speed', 'time_in_seconds', 'angular_velocity','acceleration']]
            locomotion_dfs.append(locomotion_df)
        pd.concat(locomotion_dfs).to_csv(os.path.join(locomotion_path,exp,t,'locomotion.csv'),index=False)
# %%
