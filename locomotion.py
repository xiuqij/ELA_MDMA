#%%
import os
import glob
import pandas as pd
import pickle
from multiprocessing import cpu_count, Pool, set_start_method, current_process
import utils_locomotion_and_motionless as utils

#%%
# Constants
mice =['red','blue','green','yellow']
body_part = ['1_earL','2_earR','3_nose','4_centre','5_centreL','6_centreR','7_tailBase','8_tailTip']
coord=['x','y','prob']
torso=[['4_centre_x','5_centreL_x','6_centreR_x'],['4_centre_y','5_centreL_y','6_centreR_y']]
px_mm=0.59
fps = 25
px_m=px_mm*1000

# Paths
exp = 'female_P42'
parquet_path =r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\parquet"
parquet_path_exp = os.path.join(parquet_path,exp)
nest_path=r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\nest"  
nest_exp = os.path.join(nest_path,exp)
output_path=r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\locomotion"
output_path_exp = os.path.join(output_path,exp)
time_points = ['baseline','MDMA']

#%%
def main(): #multiprocessing
    #configure_logging()

    for t in time_points:
        print(t)
        main_folder  = os.path.join(parquet_path_exp,t)
        nest_file = os.path.join(nest_exp,t,'nest.pickle')
        with open(nest_file, 'rb') as f:
            nest_dict = pickle.load(f)

        # Gather all files to process
        files_to_process = []
        for file_path in glob.glob(f"{main_folder}/*.parquet"):  
            file_basename=os.path.basename(file_path)
            file=os.path.splitext(file_basename)[0]
            
            files_to_process.append((file_path,main_folder,file,nest_dict,mice,body_part,coord,torso,px_mm,px_m,fps))
        print(len(files_to_process))
        # multiprocessing
        num_processes=cpu_count()//3 #num of cpu cores
        with Pool(processes=num_processes) as pool:
            results = pool.map(utils.processer, files_to_process)

        print("saving results...")
        composite_dfs = {}
        #composite_df=pd.DataFrame()

        # Loop over the list just once
        for item in results:
            file=item[2]
            entropy=item[1]
            df=item[0]

            composite_dfs[file]={'locomotion':df,'entropy':entropy}

        os.makedirs(os.path.join(output_path_exp,t),exist_ok=True)
        file_dict=os.path.join(output_path_exp,t,'locomotion'+'.pickle')
        with open(file_dict, "wb") as file:
            pickle.dump(composite_dfs, file)
            file.close()

        #for item in results:
        #  dataframe=item[0]
        # composite_df = pd.concat([composite_df,dataframe], ignore_index=True)
            #composite_df.to_csv(os.path.join(output_folder,f'locomotion.csv'),index=False)

if __name__ == '__main__':
    set_start_method('spawn',force=True)  # Set the start method to 'spawn'
    
    main()