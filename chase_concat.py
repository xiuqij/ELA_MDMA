#%%
import os
import pandas as pd
import glob

#%%
CHASE_FOLDER = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\chase"

EXPS = ["female_P42","male_P35"]
TPS = ["baseline","MDMA","MDMA_acute"]


#video,pair,start,end,duration,chaser,chased,box,date,timestamp
for exp in EXPS:
    dfs=[]
    for s in TPS:
        csv_path = os.path.join(CHASE_FOLDER,exp,s)
        files = glob.glob(os.path.join(csv_path,"*.csv"))
        for f in files:
            df = pd.read_csv(f)
            df['time_point'] = s
            dfs.append(df)

    df = pd.concat(dfs)
    df.to_csv(os.path.join(CHASE_FOLDER,exp,'chase_raw_events.csv'), index = False)
