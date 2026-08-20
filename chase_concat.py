#%%
import os
import pandas as pd
import glob

#%%
CHASE_FOLDER = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\July_2026\chase"

EXPS = ["female_P35","male_P42"]
TPS = ["baseline","MDMA"]


#video,pair,start,end,duration,chaser,chased,box,date,timestamp
for exp in EXPS:
    dfs=[]
    for s in TPS:
        dfs_tp = []
        csv_path = os.path.join(CHASE_FOLDER,exp,s)
        files = glob.glob(os.path.join(csv_path,"*.csv"))
        for f in files:
            df = pd.read_csv(f)
            dfs_tp.append(df)

            df['time_point'] = s
            dfs.append(df)
        df_tp = pd.concat(dfs_tp)
        df_tp.to_csv(os.path.join(CHASE_FOLDER,exp,s,'chase_events.csv'),index=False)

    df = pd.concat(dfs)
    df.to_csv(os.path.join(CHASE_FOLDER,exp,'chase_raw_events.csv'), index = False)
