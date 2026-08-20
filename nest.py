#%%
import os
import pandas as pd
import pickle
from multiprocessing import cpu_count, Pool, set_start_method
import time
import utils_nest as utils

#%%
mouse =['Red','Blue','Green','Yellow']
bodypart = ['1_earL','2_earR','3_nose','4_centre','5_centreL','6_centreR','7_tailBase','8_tailTip']
coord=['x','y','prob']
thres_lh=0.7
thres_nans=7    # 6 or 7
#thres_time=25   # changed to 25
#thres_gap=3
nest_nans=['sum_nansRed','sum_nansBlue','sum_nansGreen','sum_nansYellow']
mask_cols=['sum_nansRed_mask','sum_nansBlue_mask','sum_nansGreen_mask','sum_nansYellow_mask']
mice=['red','blue','green','yellow']
# Mouse-specific thresholds
THRESHOLDS = {
    "green":  {"time": 75, "gap": 2, "start": 20},
    "yellow": {"time": 75, "gap": 2, "start": 20},
    "red":    {"time": 75, "gap": 5, "start": 10},
    "blue":   {"time": 75, "gap": 5, "start": 10},
}
PARQUET_FOLDER = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\parquet" 
NEST_FOLDER = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\nest_try"
exp_list = ['female_P42','male_P35']   #LIST OF FOLDERS
sets = ['baseline','MDMA']

#%%
def main():
    start_time = time.time()
    for exp in exp_list:
        print(f'starting {exp}...')
        start_time_exp = time.time()
        os.makedirs(os.path.join(NEST_FOLDER,exp),exist_ok= True)
        for s in sets:
            print(f'processing {s}...')
            start_time_s = time.time()
            files_to_process =[]
            parquet_folder = os.path.join(PARQUET_FOLDER,exp,s)
            parquet_files  = sorted([f for f in os.listdir(parquet_folder) if f.endswith('.parquet')])
            print(len(parquet_files))
            for file in parquet_files:
                file_noext = os.path.splitext(file)[0]
                filepath = os.path.join(parquet_folder,file)

                files_to_process.append((filepath,parquet_folder,mouse, bodypart, coord, thres_lh, thres_nans, THRESHOLDS, nest_nans, mask_cols, mice))

            num_processes = cpu_count()//3

            with Pool(processes=num_processes) as pool:
                results = pool.map(utils.find_nest_events,files_to_process)

            print("processed.saving results...")
            composite_dfs = {}
            composite_df=pd.DataFrame()

            # Loop over the list just once
            for item in results:
                file=item[1]
                dataframe=item[0]
                composite_dfs[file]=dataframe
            
            output_folder = os.path.join(NEST_FOLDER,exp,s)
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            nest_pickle=os.path.join(output_folder,'nest.pickle')
            with open(nest_pickle, "wb") as file:
                pickle.dump(composite_dfs, file)
                file.close()

            for item in results:
                dataframe=item[0]
                composite_df = pd.concat([composite_df,dataframe], ignore_index=True)
                
            composite_df.to_csv(os.path.join(output_folder,'nest_events.csv'),index=False)
            print(f'saved {s}.')
            end_time_s = time.time()
            elapsed_time = (end_time_s-start_time_s)/60
            print(f'{s} completed in {elapsed_time} mins.')
        end_time_exp = time.time()
        print(f'{exp} completed in {(end_time_exp-start_time_exp)/60} mins.\n')
    end_time = time.time()
    print(f'All completed in {(end_time-start_time)/60} mins.')


#%%
if __name__ == '__main__':
    set_start_method('spawn',force=True)  # Set the start method to 'spawn'
    main()

# %%
