import pandas as pd 
import os
import math

def NaNs_to_1_slow(column,lik_treshold_dlc):
    NaNs_list=[0 for i in range(len(column))]
    for i in range(len(column)):
        if (column[i]<lik_treshold_dlc) or (math.isnan(column[i])) :
            NaNs_list[i]=1
        #The list contains 1 if the associated value in the column was a NaN
    return pd.Series(NaNs_list)
def NaNs_to_1(column, lik_treshold_dlc):
    return ((column < lik_treshold_dlc) | column.isna()).astype(int)
def find_bouts(mask_df, mouse_col, mouse, thresholds):
    # mouse-specific thresholds
    if mouse in thresholds:
        threshold_time = thresholds[mouse]['time']
        threshold_gap = thresholds[mouse]['gap']
        threshold_start = thresholds[mouse]['start']
    else:
        threshold_time = 25
        threshold_gap = 3
        threshold_start = 3

    # Convert once to numpy array (much faster than iloc in loops)
    arr = mask_df[mouse_col].to_numpy()

    start_frame = []
    end_frame = []

    row = threshold_time
    n = len(arr)

    while row < n:

        # Skip zeros quickly/only start with consecutive 1s
        if (
            row + threshold_start >= n 
            or not arr[row:row + threshold_start].all()
        ):
            row += 1
            continue

        # Start of event
        start = row
        start_frame.append(start)

        count = row

        # Extend bout while there is at least one "1"
        # within the next threshold_gap frames
        while (
            count + 1 < n
            and arr[count + 1 : min(count + 1 + threshold_gap, n)].any()
        ):
            count += 1

            # Optional debugging for very long bouts
            # if count % 50000 == 0:
            #     print(f"{mouse}: count={count}", flush=True)

        end_frame.append(count)

        # Move to next position after bout
        row = count + 1

    # Build dataframe
    frames = [
        [i, j]
        for i, j in zip(start_frame, end_frame)
        if (j - i) >= threshold_time
    ]

    frames = pd.DataFrame(frames, columns=['start_frame', 'end_frame'])

    if len(frames) > 0:
        frames['duration'] = frames['end_frame'] - frames['start_frame']
        frames['mouse'] = mouse
    else:
        frames = pd.DataFrame(
            columns=['start_frame', 'end_frame', 'duration', 'mouse']
        )

    return frames

def find_nest_events(info):
    parquet_file,main_folder,mouse,body_part,coord,lik_treshold_dlc,threshold_nans,thresholds_dict,nest_nans,mask_cols,mice=info

    #print(f"START {os.path.basename(parquet_file)}", flush=True)
    pq_file_path =parquet_file
    df=pd.read_parquet(pq_file_path, engine='pyarrow')
    #print(f"LOADED {os.path.basename(parquet_file)}", flush=True)
    
    file=os.path.basename(pq_file_path)
    file=os.path.splitext(file)[0] #2024-12-17_16-44-02_crop_green_corrected
    
    col_names = [(item1+'_'+item2+'_'+ item3) for item1 in mouse for item2 in body_part for item3 in coord]
    df.columns = col_names
    lik_cols=[col for col in df.columns if 'prob' in col]
    nans_df=pd.DataFrame()
    for col in lik_cols:
        nans_df[col]=NaNs_to_1(df.loc[:,col],lik_treshold_dlc)
    nans_df.columns = lik_cols
    counter=8
    for m in mouse:
        sub_df=nans_df.iloc[:,counter-8:counter]
        nans_df['sum_nans'+m]=sub_df.sum(axis=1)
        counter+=8

    mask_df=pd.DataFrame()
    for nan_col in nest_nans:
        mask_col = nan_col + '_mask'
        mask_df[mask_col] = (nans_df[nan_col] >= threshold_nans).astype(int)

    nest_df=pd.DataFrame()

    for mask_col,mouse_name in zip(mask_cols,mice):
        events=find_bouts(mask_df, mouse_col=mask_col,mouse=mouse_name,thresholds=thresholds_dict)
        nest_df=pd.concat([nest_df,events])


    nest_df['video']=file
    nest_df['box'] = file.split('_')[3]
    nest_df['date'] = file.split('_')[0]
    nest_df['timestamp'] = file.split('_')[1]

    nest_df=nest_df.reset_index(drop=True)  
    nest_df = nest_df[['video','mouse','start_frame', 'end_frame', 'duration','box','date','timestamp']]
    #print(f"DONE {os.path.basename(parquet_file)}", flush=True)
    return (nest_df,file)

