#%%
import os
import glob
import pandas as pd
import numpy as np
import pickle
from multiprocessing import cpu_count, Pool, set_start_method, current_process
import utils_locomotion_and_motionless as utils

#%%
# Define your constants and thresholds
main_parts = ['1_earL', '2_earR', '3_nose', '4_centre', '5_centreL', '6_centreR', '7_tailBase', '8_tailTip']
mice = ['red', 'blue', 'green', 'yellow']
coord = ['x', 'y', 'prob']

fps = 25
likelihood_threshold = 0.5

# extra functions
def project_centroid_position(df, mouse):
    centroid_x1 = 0
    centroid_y1 = 0
    try:
        centroid_x1 = df[mouse + '_4_centre' + '_x'].where(df[mouse + '_4_centre' + '_prob'] > likelihood_threshold,
                                                           (df[mouse + '_1_earL' + '_x'] + df[mouse + '_2_earR' + '_x']) / 2)

        centroid_y1 = df[mouse + '_4_centre' + '_y'].where(df[mouse + '_4_centre' + '_prob'] > likelihood_threshold,
                                                           (df[mouse + '_1_earL' + '_y'] + df[mouse + '_2_earR' + '_y']) / 2)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return centroid_x1, centroid_y1

def process_file(file_path):
    pq_file = os.path.basename(file_path)
    df = pd.read_parquet(file_path, engine='pyarrow')
    col_names = [(item1 + '_' + item2 + '_' + item3) for item1 in mice for item2 in main_parts for item3 in coord]
    df.columns = col_names

    all_events = []
    all_events_speeding = []
    print("Analysing file   " + pq_file)
    for mouse in mice:
        features_df = pd.DataFrame(index=df.index)
        centroid_x1, centroid_y1 = project_centroid_position(df, mouse)
        features_df['torso_x'] = centroid_x1
        features_df['torso_y'] = centroid_y1
        features_df['delta_x'] = features_df['torso_x'].diff()
        features_df['delta_y'] = features_df['torso_y'].diff()
        features_df['velocity'] = np.sqrt(features_df['delta_x'] ** 2 + features_df['delta_y'] ** 2)

        features_df['visible_parts'] = 0
        for bodypart in main_parts:
            features_df['bodypart_visible'] = df[mouse + '_' + bodypart + '_prob'] >= 0.5
            features_df['visible_parts'] += features_df['bodypart_visible'].astype(int)

        percentile_speeding = features_df['velocity'].quantile(0.95)
        features_df['speeding'] = (features_df['velocity'].gt(percentile_speeding)) & (features_df['visible_parts'] >= 3)

        percentile_motionless = features_df['velocity'].quantile(0.15)
        features_df['motionless'] = (features_df['velocity'].lt(percentile_motionless)) & (features_df['visible_parts'] >= 3)

        speeding_events = utils.find_events(features_df['speeding'], min_event_duration=50, max_gap=10)
        motionless_events = utils.find_events(features_df['motionless'], min_event_duration=50, max_gap=10)

        events_df = pd.DataFrame(motionless_events, columns=['start_frame', 'end_frame'])
        events_df['duration'] = events_df['end_frame'] - events_df['start_frame'] + 1
        events_df['mouse'] = mouse
        events_df['video'] = os.path.splitext(pq_file)[0]
        all_events.append(events_df)

        speeding_events_df = pd.DataFrame(speeding_events, columns=['start_frame', 'end_frame'])
        speeding_events_df['duration'] = speeding_events_df['end_frame'] - speeding_events_df['start_frame'] + 1
        speeding_events_df['mouse'] = mouse
        speeding_events_df['video'] = os.path.splitext(pq_file)[0]
        all_events_speeding.append(speeding_events_df)

    return (all_events, all_events_speeding)

#%%
# paths
parquet_path = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\parquet"
output_path = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\immobility"

exp = 'male_P35'
time_points = ['baseline','MDMA']
parquet_path_exp = os.path.join(parquet_path, exp)
output_path_exp = os.path.join(output_path, exp)
#%%
def main():
    #start_time = time.time()
    #warnings.filterwarnings("ignore", category=DeprecationWarning)
    for t in time_points:
        print(t)
        main_folder  = os.path.join(parquet_path_exp,t)
        files_to_process = glob.glob(f"{main_folder}/*.parquet")
        print(len(files_to_process))
        batch_size = 10 # Adjust this based on your memory constraints
        full_event_data = pd.DataFrame()
        full_event_data_speeding = pd.DataFrame()
        
        for i in range(0, len(files_to_process), batch_size):
            batch_files = files_to_process[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1} of {len(files_to_process) // batch_size + 1}")

            with Pool() as pool:
                results = pool.map(process_file, batch_files)

            for events_pairs in results:
                for pair_df in events_pairs[0]:
                    full_event_data = pd.concat((full_event_data, pair_df), ignore_index=True)
                for pair_df in events_pairs[1]:
                    full_event_data_speeding = pd.concat((full_event_data_speeding, pair_df), ignore_index=True)
        print("saving results...")
        os.makedirs(os.path.join(output_path_exp, t), exist_ok=True)
        full_event_data.to_csv(os.path.join(output_path_exp, t, 'motionless_events.csv'), index=False)
        full_event_data_speeding.to_csv(os.path.join(output_path_exp, t, 'speeding_events.csv'), index=False)
if __name__ == '__main__':
    set_start_method('spawn')  # Set the start method to 'spawn'
    main()