#%%
import os
import pandas as pd
import numpy as np
import glob
import utils_proximity as utils
from multiprocessing import cpu_count, Pool, set_start_method
import time
import pickle


#%%
def main(): #multiprocessing
    exp_list = ['female_P42','male_P35']
    tp = ['baseline','MDMA','MDMA_acute']
    for exp in exp_list:
        print(f'Start processing {exp}')
        start = time.time()
        for t in tp:
            print(t)
            parquet_folder = f"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\parquet\{exp}\{t}"
            output_folder = f"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\proximity\{exp}\{t}"

            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            # modify to define extension area as a percentage of the average mouse area
            px_to_mm=0.59
            extension_area_px=18/px_to_mm 
            pairs={'BG':[i for i in range(24,72)],'BY':[i for i in range(24,48)]+[i for i in range(72,96)],'GY':[i for i in range(48,96)],'RB':[i for i in range(0,48)],'RG':[i for i in range(0,24)]+[i for i in range(48,72)],'RY':[i for i in range(0,24)]+[i for i in range(72,96)]}
            thres_time = 4
            thres_gap = 8

            # chase files
            chase_path = f"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\chase\{exp}\chase_raw_events.csv"
            chase_df = pd.read_csv(chase_path)

            # Get inputs
            files_to_process = []
            for filepath in glob.glob(f'{parquet_folder}/*.parquet'):
                files_to_process.append((filepath,parquet_folder,pairs,thres_time,thres_gap,extension_area_px,px_to_mm))

            # checkpoint
            print(f"Found {len(files_to_process)} parquet files")
            
            # set multiprocessing
            num_processes=cpu_count()//3 #num of cpu cores

            with Pool(processes=num_processes) as pool:
                results = pool.map(utils.processer, files_to_process)

            composite_dfs = {}

            # Loop over the list just once
            for item in results:
                file=item[1]
                dictionary=item[0]
                composite_dfs[file]={}
                for key, dfs in dictionary.items():
                    if key not in composite_dfs:
                        composite_dfs[file][key] = []
                    composite_dfs[file][key].extend(dfs)

            # save unfiltered result
            print("saving results (unfiltered).")
            file_dict=os.path.join(output_folder,f'proximity_dict_{exp}_{t}.pickle')
            with open(file_dict, "wb") as file:
                pickle.dump(composite_dfs, file)
                file.close()

            #exclude chases and recalculate bouts
            print("saving results (filtered out chases).")
            dict_predicted_no_chases=utils.exclude_chases(chase_df,composite_dfs,thres_time,thres_gap)
            file_dict_no_chases=os.path.join(output_folder,f'proximity_dict_{exp}_no_chases.pickle')
            with open(file_dict_no_chases, "wb") as file:
                pickle.dump(dict_predicted_no_chases, file)
                file.close()
        stop = time.time()
        print(f'{exp} completed in {(stop-start)/60} min.\n')


#%%
if __name__ == '__main__':
    set_start_method('spawn',force=True)  # Set the start method to 'spawn'
    main()
# %%