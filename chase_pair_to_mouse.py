#%%
import os
import pandas as pd


#%%
CHASE_FOLDER = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\chase"
EXPS = ["female_P42","male_P35"]
#%% Mouse level
def pair_to_mouse(df):
    cols = ["mouse", "video", "time_point", "box", "date", "timestamp", "start", "end", "frames_chasing", "frames_chased", "event_chasing", "event_chased"]
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
            'time_point': r['time_point'],
            'box': r['box'],
            'date': r['date'],
            'timestamp': r['timestamp'],
            'start': r['start'],
            'end': r['end'],
            'frames_chasing': duration,
            'frames_chased': 0,
            'event_chasing': 1,
            'event_chased': 0
        })

        # chased row
        rows.append({
            'mouse': r['chased'],
            'video': r['video'],
            'time_point': r['time_point'],
            'box': r['box'],
            'date': r['date'],
            'timestamp': r['timestamp'],
            'start': r['start'],
            'end': r['end'],
            'frames_chasing': 0,
            'frames_chased': duration,
            'event_chasing': 0,
            'event_chased': 1
        })

    return pd.DataFrame(rows,columns=cols)


for exp in EXPS:
    df_pair = pd.read_csv(os.path.join(CHASE_FOLDER,exp,'chase_raw_events.csv'))
    df_mouse = pair_to_mouse(df_pair)
    df_mouse.to_csv(os.path.join(CHASE_FOLDER,exp,'chase_events_by_mouse.csv'),index=False)

