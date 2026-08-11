import pandas as pd
import numpy as np
import pickle
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
from scipy.stats import entropy
import os
import glob
from multiprocessing import cpu_count, Pool, set_start_method
from time import sleep
import logging

#auxiliary function to calculate centroids
def calculate_centroid(x_values,y_values):
    x_centroid = np.nanmean(x_values)
    y_centroid = np.nanmean(y_values)
    return x_centroid, y_centroid


def processer(info):
    file_path,main_folder,file,nest_dict,mice,body_part,coord,torso,px_mm,px_m,fps=info
    distance_df=pd.DataFrame()
    nest_df=nest_dict[file] #from the nest classifier

    #dictionary in format {red:[],yellow:[]...} to store a list of frames where the mouse is known to be in the nest, for each mouse
    nest_frames_mice={}
    for mouse in mice:
        mouse_nest_df=nest_df[nest_df['mouse']==mouse]
        nest_frames = [num for j in mouse_nest_df.index for num in range(mouse_nest_df.loc[j, 'start_frame'], mouse_nest_df.loc[j, 'end_frame'] + 1)]
        nest_frames_mice[mouse]=nest_frames

    df=pd.read_parquet(file_path,engine='pyarrow')
    col_names = [(item1+'_'+item2+'_'+ item3) for item1 in mice for item2 in body_part for item3 in coord]
    df.columns = col_names

    entropy_dict={}

    for mouse in mice:
        centroids = []
        frame_indices = []  # Store frame indices for each pair of centroids
        for i in range(0, len(df), fps):  # Iterate every fps frames 
            if i in nest_frames_mice[mouse]:  #centroids will not be calculated for nest frames, I don't want to include them in the locomotion calculations
                continue
                
            x_points = [f"{mouse}_{x}" for x in torso[0]] #proper columns for that mouse and bps
            y_points = [f"{mouse}_{y}" for y in torso[1]] #proper columns for that mouse and bps
            x_values = df.loc[i, x_points]
            y_values = df.loc[i, y_points]

            centroid = calculate_centroid(x_values, y_values)

            if np.isnan(centroid).any() & len(frame_indices)>5: #if the centroid is nan, and not in initial frames
                centroid=np.mean(centroids[len(frame_indices)-5:len(frame_indices)-1],axis=0)  #centroid should be the mean of 5 previous centroids
            
            centroids.append(np.array(centroid))  # Convert tuple to numpy array
            frame_indices.append(i)     #keep track of the frames
    
    
        # Compute Euclidean distance between centroids fps frames apart
        distances = [np.linalg.norm(centroids[j] - centroids[j+1]) for j in range(len(centroids)-1)]
        distances=[dist/px_m for dist in distances]  #convert to meters

        #edge case
        if len(distances)==0:
            distances=[0]
            time_seconds=[np.nan]
            velocities=[np.nan]
            adjusted_angular_velocities=[np.nan]
            accelerations=[np.nan]
        
        else:
            frame_indices = frame_indices[:len(distances)] # Adjust frame indices length 
            time_seconds=  [idx/fps for idx in frame_indices]

            distances = [distances[dist] if dist == 0 or (time_seconds[dist] - time_seconds[dist-1] == 1) else np.nan for dist in range(len(distances))]  #distance is nan when the mouse was in the nest

            times = np.diff(time_seconds)
            times=np.concatenate([[1],times])
            velocities = [distance / time for distance, time in zip(distances, times)]
            if len(velocities)<=1:
                logging.warning(f"len vel: {velocities}, file: {file}, mouse: {mouse}")

            if len(centroids)>0:
                dx=np.diff([cx[0] for cx in centroids])
                dy=np.diff([cy[1] for cy in centroids])
                angles = [np.arctan2(x,y) for x,y in zip(dx,dy)]
                dtheta = np.diff(angles)
                angular_velocities = [dt / t for dt, t in zip(dtheta,times)]
                angular_velocities=np.concatenate([[np.nan],angular_velocities])

            else:
            
                dx=[]
                dy=[]
                angles=[]
                angular_velocities=[]


    # to put nans where mouse in nest (diff(time)>1s)
            adjusted_angular_velocities = [angular_velocities[0]]  # Initial value remains the same

            for i in range(1,len(times)):
                if times[i] == 1:
                    adjusted_angular_velocities.append(angular_velocities[i])
                else:
                    adjusted_angular_velocities.append(np.nan)

            accelerations = np.diff(velocities) / times[:-1]
        
            try:
                accelerations = np.concatenate([[velocities[0]], accelerations])  # Adjust length to match velocities
            except IndexError: 
            
                accelerations=[]
        
        # Create DataFrame for current mouse

            logging.warning(f"index error: {len(distances),len(velocities),len(accelerations),len(adjusted_angular_velocities),len(time_seconds)}, file: {file}, mouse: {mouse}")
        mouse_df = pd.DataFrame({'mouse': [mouse] * (len(distances)),
                                'distance': distances,
                                'speed':velocities,
                                'time_in_seconds': time_seconds,
                                'angular_velocity': adjusted_angular_velocities,
                                'acceleration':accelerations})
        
        cleaned_angular_velocities = [x for x in adjusted_angular_velocities if not pd.isna(x)]
        prob_distribution, _ = np.histogram(cleaned_angular_velocities, bins='auto', density=True)
        trajectory_entropy = entropy(prob_distribution)
        entropy_dict[mouse]=trajectory_entropy

        # Append DataFrame for current mouse to distance_df
        distance_df = pd.concat([distance_df, mouse_df], ignore_index=True)


    return distance_df,entropy_dict,file

# from motionless_batch_processing
def find_events(binary_series, min_event_duration=1, max_gap=25):

    # Function to calculate the length of an event
    def event_length(event):
        return event[1] - event[0] + 1

    # Detect raw events
    events = []
    start_index = None
    for i, value in enumerate(binary_series):
        if value == 1 and start_index is None:
            start_index = i
        elif value == 0 and start_index is not None:
            events.append((start_index, i - 1))
            start_index = None
    if start_index is not None:
        events.append((start_index, len(binary_series) - 1))

    # Merge events based on max_gap and relative event sizes
    merged_events = []
    prev_event = None
    for event in events:
        if prev_event is None:
            prev_event = event
        else:
            gap_length = event[0] - prev_event[1] - 1
            if gap_length <= max_gap and gap_length < event_length(prev_event):
                prev_event = (prev_event[0], event[1])  # Merge events
            else:
                merged_events.append(prev_event)
                prev_event = event
    if prev_event:
        merged_events.append(prev_event)

    # Filter events that do not meet the min_event_duration criteria
    final_events = [event for event in merged_events if event_length(event) >= min_event_duration]

    return final_events



#merges the videosplits and adds an extra column defining the timebin 
def merge_and_timebins(df, videosplit_duration, time_window_size, total_experiment_duration):
    new_rows = []

    for exp in df['exp'].unique():
        for day in df['day'].unique():
            # Combine the video splits
            condition = (df['videosplit'] == 2) & (df['day'] == day) & (df['exp'] == exp)
            df.loc[condition, 'start_frame'] += videosplit_duration
            df.loc[condition, 'end_frame'] += videosplit_duration

            # Assign time bins
            df['time_bin'] = ((df['start_frame'] % total_experiment_duration) // time_window_size).astype(int) + 1

            for _, row in df.iterrows():
                timebin_frames = row['time_bin'] * time_window_size
                while row['end_frame'] > timebin_frames:
                    new_row = row.copy()
                    new_row['start_frame'] = timebin_frames + 1
                    new_row['end_frame'] = min(row['end_frame'], (new_row['time_bin'] + 1) * time_window_size)
                    new_row['time_bin'] += 1
                    new_row['duration'] = new_row['end_frame'] - new_row['start_frame'] + 1
                    new_rows.append(new_row)
                    df.at[_, 'end_frame'] = timebin_frames
                    df.at[_, 'duration'] = df.at[_, 'end_frame'] - df.at[_, 'start_frame'] + 1
                    timebin_frames += time_window_size

    new_df = pd.concat([df, pd.DataFrame(new_rows)])
    return new_df



def merge_and_timebins(df, videosplit_duration, time_window_size, total_experiment_duration):
    new_rows = []

    for exp in df['exp'].unique():
        for day in df['day'].unique():
            # Combine the video splits
            condition = (df['videosplit'] == 2) & (df['day'] == day) & (df['exp'] == exp)
            df.loc[condition, 'start_frame'] += videosplit_duration
            df.loc[condition, 'end_frame'] += videosplit_duration

            # Assign time bins
            df['time_bin'] = ((df['start_frame'] % total_experiment_duration) // time_window_size).astype(int) + 1

            for _, row in df.iterrows():
                timebin_frames = row['time_bin'] * time_window_size
                while row['end_frame'] > timebin_frames:
                    new_row = row.copy()
                    new_row['start_frame'] = timebin_frames + 1
                    new_row['end_frame'] = min(row['end_frame'], (new_row['time_bin'] + 1) * time_window_size)
                    new_row['time_bin'] += 1
                    new_row['duration'] = new_row['end_frame'] - new_row['start_frame'] + 1
                    new_rows.append(new_row)
                    df.at[_, 'end_frame'] = timebin_frames
                    df.at[_, 'duration'] = df.at[_, 'end_frame'] - df.at[_, 'start_frame'] + 1
                    timebin_frames += time_window_size

    new_df = pd.concat([df, pd.DataFrame(new_rows)])
    return new_df




 






