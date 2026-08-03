#%%
import pandas as pd
import os
import multiprocessing
from multiprocessing import Pool, set_start_method, pool
import time
import utils_ROIs as utils
#import importlib
#importlib.reload(utils)

#%%
PARQUET_FOLDER = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\parquet" 
ROI_FOLDER = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\ROIs"
exp_list = ['male_P35','female_P42']
sets = ['baseline','MDMA_acute','MDMA']
#%%
def main():
    # Start the timer
    start_time = time.time()
    # process one folder at a time
    for exp in exp_list:
        start_time_exp = time.time()
        print(f'Processing... {exp}\n')
        os.makedirs(os.path.join(ROI_FOLDER,exp),exist_ok = True)
        for s in sets:
            start_time_set = time.time()
            print(f'Processing... {s}\n')
            results={}
            arenas_roi = {}
            # Load the data
            parquet_path = os.path.join(PARQUET_FOLDER,exp,s) # change to the folder where parquet files are
            csv_path = os.path.join(ROI_FOLDER,exp,s) # change to your results folder
            if not os.path.exists(csv_path):
                os.makedirs(csv_path)
            roi_names = ['s_wall','above_nest','ramp1','ramp2','non_wall','woodstick','feeder_prox', 'feeder_dist','water_prox','water_dist']
            # check how you saved the roi info:
            roi_dict = pd.read_pickle(os.path.join(ROI_FOLDER,f"{exp}.pickle"))
            # Gather all files to process
            files_to_process = []
            for file in os.listdir(parquet_path):
                #Example file name: 2024-12-17_16-44-02_crop_green.parquet
                file_path = os.path.join(parquet_path,file)
                video_key = os.path.splitext(file)[0]
                video_box = video_key.split("_")[3]    # Get the color of box
                video_roi = roi_dict[video_box] # Get the corresponding ROI coords of the box box

                arenas_roi[video_key] = utils.ROIS(video_roi,roi_names,video_key)
                files_to_process.append((file_path,file_path,arenas_roi))
            
            num_processes = multiprocessing.cpu_count()//3  #num of cpu cores

            with Pool(processes=num_processes) as pool:
                results= pool.map(utils.process_file,files_to_process)    #chunksize = len(files_to_process)//num_processes
            
            # Initialize a dictionary to hold lists of DataFrames for each key
            composite_dfs = {}

            # Loop over the list just once
            for item in results:
                for key, dfs in item.items():
                    if key not in composite_dfs:
                        composite_dfs[key] = []
                    composite_dfs[key].extend(dfs)

            # Concatenate and export each list of DataFrames to a .csv file
            for key, dfs in composite_dfs.items():
                composite_df = pd.concat(dfs, ignore_index=True)
                composite_df.to_csv(os.path.join(csv_path,f'{key}_events.csv'), index=False)

            end_time_set = time.time()
            # Calculate the difference
            elapsed_time = (end_time_set - start_time_set) / 3600
            print(f"Folder {s} completed in {elapsed_time} hours.")
        end_time_exp = time.time()
        print(f"Folder {exp} completed in {(end_time_exp - start_time_exp)/3600} hours.")
    end_time = time.time()
    print(f"All completed in {(end_time - start_time)/3600} hours.")
#%%
if __name__ == '__main__':
    set_start_method('spawn',force=True)  # Set the start method to 'spawn'
    main()



# %%