#%%
import os
import pandas as pd
import numpy as np
import utils_stats as utils_stats
import utils_hierarchies as utils

#%%
'''
workflow:
use 12h as the smallest unit for hiearchy
1. build win-loss matrix from raw chase events (box x day)
2. convert wins to proportions
3. calculate DS and normDS
4. assign ranks
'''

#%%
def compute_hierarchy(df,mice = ['red','blue','yellow','green']):
    # day, phase, box, mouse, normDS, rank
    results = []
    for (box, day, phase),subdf in df.groupby(["box","day","phase"]):
        

        wl_mat, idx_to_mouse = utils.build_wl_matrix(
            subdf,
            mice
        )

        DS, normDS = utils.davids_score_from_matrix(wl_mat)
        order = np.argsort(-normDS)
        print(order)

        for rank_pos, mouse_idx in enumerate(order):
            results.append({
                "box": box,
                "day": day,
                "phase": phase,
                "mouse": idx_to_mouse[mouse_idx],
                "normDS": normDS[mouse_idx],
                "rank": ["Alpha", "Beta", "Gamma", "Delta"][rank_pos]
            })
    hierarchy_df = pd.DataFrame(results)
    return hierarchy_df
#%%
# PATHS
main_path = "/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA"
batches = ['April','July']
exps = {'April':['male_P35','female_P42'],'July':['female_P35','male_P42']}
time_points = ['baseline','MDMA']

#%%
cols = ['day','phase','box','pair','chaser','chased','start','end','duration']
for b in batches:
    print(b)
    path_chase = os.path.join(main_path,f'{b}_2026','chase')
    for exp in exps[b]:
        print(exp)
        for t in time_points:
            print(t)
            df = pd.read_csv(os.path.join(path_chase,exp,t,'chase_events.csv'))
            df = utils_stats.compute_event_times(df, start_col='start',end_col='end')
            df = utils_stats.add_time_labels(df)
            df = utils_stats.add_day_order(df)
            mice_list = ['red','blue','yellow','green']
            if b=='April' and exp=='male_P35' and t=='MDMA':
                df['day'] = df['day'] + 1
            df = df [cols]
            hierarchy_df = compute_hierarchy(df,mice=mice_list)
            hierarchy_df.to_csv(os.path.join(path_chase,exp,t,'hierarchy.csv'),index=False)
            
#%%
# re-run for July/male_P42/MDMA/box8
cols = ['day','phase','box','pair','chaser','chased','start','end','duration']

df = pd.read_csv(os.path.join('/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA/July_2026/chase/male_P42/MDMA','chase_events.csv'))
df = utils_stats.compute_event_times(df, start_col='start',end_col='end')
df = utils_stats.add_time_labels(df)
df = utils_stats.add_day_order(df)
df = df [cols]

results = []
for (box, day, phase),subdf in df.groupby(["box","day","phase"]):
    mice = ['red','yellow','green'] if box ==8 else ['red','blue','yellow','green']
    wl_mat, idx_to_mouse = utils.build_wl_matrix(
            subdf,
            mice
        )

    DS, normDS = utils.davids_score_from_matrix(wl_mat)
    order = np.argsort(-normDS)
    print(order)

    for rank_pos, mouse_idx in enumerate(order):
        results.append({
            "box": box,
            "day": day,
            "phase": phase,
            "mouse": idx_to_mouse[mouse_idx],
            "normDS": normDS[mouse_idx],
            "rank": ["Alpha", "Beta", "Gamma", "Delta"][rank_pos]
            })
    hierarchy_df = pd.DataFrame(results)

hierarchy_df.to_csv(os.path.join('/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA/July_2026/chase/male_P42/MDMA','hierarchy.csv'),index=False)

# %%
