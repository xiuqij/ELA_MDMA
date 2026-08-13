#%%
import os
import pandas as pd

#%%
def csv_to_parquet(csv_path,parquet_path):
    '''with a given csv file path, read in the dataframe and save as parquet'''
    df = pd.read_csv(csv_path,skiprows=[1,2,3],low_memory=False)
    df = df.drop('scorer',axis=1,errors='ignore')
    df.to_parquet(parquet_path)
#%% 
def get_csv_files(folder):
    '''get all csv files in a given folder'''
    csv_files = [f for f in os.listdir(folder) if f.endswith('filtered.csv')]
    return csv_files



#%% Define input and output folders
video_path = "/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA/July_2026/cropped_videos"
parquet_path = "/Volumes/labs/Lopez Laboratory - NEURO/Xiuqi/ELA_MDMA/July_2026/parquet"

# %% 
exps = ["female_P35"]
for exp in exps:
    print(f'Processing folder: {exp}\n')
    input_path = os.path.join(video_path,exp)
    output_path = os.path.join(parquet_path,exp)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for csv_file in get_csv_files(input_path):
        print(f'Processing.. {csv_file}\n')
        #2024-12-17_04-45-46_crop_red_correctedDLC_resnet50_Sexy16PilotMar10shuffle9_180000_el_filtered.csv
        name = os.path.splitext(csv_file)[0].split('DLC')[0]
        save_path = os.path.join(output_path,f'{name}.parquet')
        csv_to_parquet(csv_path=os.path.join(input_path,csv_file),parquet_path=save_path)

        print(f'Saved to {save_path}.\n')

# %%
