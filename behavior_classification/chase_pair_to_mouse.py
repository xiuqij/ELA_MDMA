#%%
import os
import pandas as pd

#%% Mouse level
def pair_to_mouse(df):
    # df should have: video,pair,start,end,duration,chaser,chased,box,date,timestamp,time_point

    cols = ["mouse", "video", "box", "date", "timestamp", "start_frame", "end_frame", "frames_chasing", "frames_chased", "count_chasing", "count_chased"]
    rows = []
    for _, r in df.iterrows():
        duration = r['duration']

        # skip undetermined (important)
        if r['chaser'] == 'undetermined' or r['chased'] == 'undetermined':
            continue
        # chaser row
        rows.append({
            'mouse': r['chaser'],
            'video': r['video'],
            #'time_point': r['time_point'],
            'box': r['box'],
            'date': r['date'],
            'timestamp': r['timestamp'],
            'start_frame': r['start'],
            'end_frame': r['end'],
            'frames_chasing': duration,
            'frames_chased': 0,
            'count_chasing': 1,
            'count_chased': 0
        })

        # chased row
        rows.append({
            'mouse': r['chased'],
            'video': r['video'],
            #'time_point': r['time_point'],
            'box': r['box'],
            'date': r['date'],
            'timestamp': r['timestamp'],
            'start_frame': r['start'],
            'end_frame': r['end'],
            'frames_chasing': 0,
            'frames_chased': duration,
            'count_chasing': 0,
            'count_chased': 1
        })

    return pd.DataFrame(rows,columns=cols)

#%%
# Define paths
main_path = "/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA"
batches = ['April','July']
exps = {'April':['male_P35','female_P42'],'July':['female_P35','male_P42']}
time_points = ['baseline','MDMA']

for b in batches:
    print(b)
    path_chase = os.path.join(main_path,f'{b}_2026','chase')
    for exp in exps[b]:
        print(exp)
        for t in time_points:
            print(t)
            df_pair = pd.read_csv(os.path.join(path_chase,exp,t,'chase_events.csv'))
            df_mouse = pair_to_mouse(df_pair)
            df_mouse.to_csv(os.path.join(path_chase,exp,t,'chase_events_by_mouse.csv'),index=False)


# %%
