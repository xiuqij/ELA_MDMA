#%%
import os
import cv2
import pandas as pd
import math

#%%
def rename_columns(df, 
                   ANIMALS = ['Red','Blue','Green','Yellow'],
                   BPS = ['1_earL','2_earR','3_nose','4_centre','5_centreL','6_centreR','7_tailBase','8_tailTip'],
                   COORDS = ['x', 'y', 'likelihood']):
    expected_cols =[]
    for animal in ANIMALS:
        for bp in BPS:
            for coord in COORDS:
                expected_cols.append(f'{animal}_{bp}_{coord}')
    if len(expected_cols) != len(df.columns):
        raise ValueError(
            f'Column mismatch! Expected {len(expected_cols)}, got {len(df.columns)}'
        )
    df.columns = expected_cols
    

# %% Export video clips
def export_clips_single(events_df, video_path,path_dlc, video_export_path, fps=25, buffer_frames=0):
    '''Adapted script from Daniil. 
    This function export video clips from a given video with a given event dataframe.'''
    # Read in the DLC output
    df_dlc = pd.read_parquet(path_dlc)
    rename_columns(df_dlc)
    # Make output folder if not there already
    if not os.path.exists(video_export_path):
        os.makedirs(video_export_path)
    # Make clips for each event and save to output folder
    for index, row in events_df.iterrows():
        mouse = row['mouse']
        start_frame = max(0, row['start_frame'] - buffer_frames)
        end_frame = row['end_frame'] + buffer_frames

        bodyparts = ['1_earL','2_earR','3_nose','4_centre','5_centreL','6_centreR','7_tailBase','8_tailTip']
        mouse_dict = {'blue':'Blue','yellow':'Yellow','red':'Red','green':'Green'}
        individual = mouse_dict[mouse] 

        clip_folder = f'predictions_{mouse}'
        clip_path = os.path.join(video_export_path,clip_folder)
        if not os.path.exists(clip_path):
            os.makedirs(clip_path)
        clip_name = f'{index}_{mouse}_{start_frame}_{end_frame}.mp4'
        clip_filepath = os.path.join(clip_path,clip_name)
        print(f'saving... {clip_filepath}\n')

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
        out = cv2.VideoWriter(clip_filepath, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            row = df_dlc.loc[current_frame]

            for part in bodyparts:
                x = row[f'{individual}_{part}_x']
                y = row[f'{individual}_{part}_y']

                if (not math.isnan(x)) & (not math.isnan(y)):
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
            
            current_frame += 1
            out.write(frame)
        cap.release()
        out.release()




#%% Make clips for nest
src_folder = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\cropped_videos\female_P42"
pq_folder=r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\parquet\female_P42\baseline"
dest_folder = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\validation\nest_try"
video_names = ['2026-04-11_19-30-20_crop_5','2026-04-11_19-30-20_crop_6','2026-04-11_19-30-20_crop_7','2026-04-11_19-30-20_crop_8',
               '2026-04-11_19-30-22_crop_1','2026-04-11_19-30-22_crop_2','2026-04-11_19-30-22_crop_3','2026-04-11_19-30-22_crop_4']

event_df_path = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\nest_try\female_P42\baseline\nest_events.csv"
for video in video_names:
    video_filepath = os.path.join(src_folder,f'{video}.mp4')
    dlc_filepath = os.path.join(pq_folder,f'{video}.parquet')

    event_df_all = pd.read_csv(event_df_path)
    event_df = event_df_all[event_df_all['video']==video]
    event_df_clips = event_df[(event_df['start_frame']>=10000) & (event_df['end_frame']<=60000)]
    event_df_clips[['video','mouse','start_frame','end_frame','duration']].to_excel(os.path.join(dest_folder,f'validation_{video}.xlsx'))

    output_folder = os.path.join(dest_folder,video)
    os.makedirs(output_folder, exist_ok = True)
    
    export_clips_single(events_df = event_df_clips, video_path = video_filepath, path_dlc = dlc_filepath, video_export_path = output_folder)

#%% Make clips for nest
src_folder = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\cropped_videos\male_P35"
pq_folder=r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\parquet\male_P35\baseline"
dest_folder = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\validation\nest_try"
video_names = ['2026-04-04_19-26-02_crop_5','2026-04-04_19-26-02_crop_6','2026-04-04_19-26-02_crop_7','2026-04-04_19-26-02_crop_8',
               '2026-04-04_19-26-05_crop_1','2026-04-04_19-26-05_crop_2','2026-04-04_19-26-05_crop_3','2026-04-04_19-26-05_crop_4']

event_df_path = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\nest_try\male_P35\baseline\nest_events.csv"
for video in video_names:
    video_filepath = os.path.join(src_folder,f'{video}.mp4')
    dlc_filepath = os.path.join(pq_folder,f'{video}.parquet')

    event_df_all = pd.read_csv(event_df_path)
    event_df = event_df_all[event_df_all['video']==video]
    event_df_clips = event_df[(event_df['start_frame']>=10000) & (event_df['end_frame']<=60000)]
    event_df_clips[['video','mouse','start_frame','end_frame','duration']].to_excel(os.path.join(dest_folder,f'validation_{video}.xlsx'))

    output_folder = os.path.join(dest_folder,video)
    os.makedirs(output_folder, exist_ok = True)
    
    export_clips_single(events_df = event_df_clips, video_path = video_filepath, path_dlc = dlc_filepath, video_export_path = output_folder)



# %% ROIs
src_folder = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\July_2026\cropped_videos\female_P35"
pq_folder=r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\July_2026\parquet\female_P35\baseline"
dest_folder = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\July_2026\validation\ROI"
video_names = ['2026-07-24_17-55-39_crop_5','2026-07-24_17-55-39_crop_6','2026-07-24_17-55-39_crop_7','2026-07-24_17-55-39_crop_8',
               '2026-07-26_01-56-02_crop_1','2026-07-26_01-56-02_crop_2','2026-07-26_01-56-02_crop_3','2026-07-26_01-56-02_crop_4']

ROIs = ['s_wall','ramp1','ramp2','woodstick','feeder_prox', 'feeder_dist','water_prox','water_dist']
events_folder = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\July_2026\ROIs\female_P35\baseline"

for video in video_names:
    video_filepath = os.path.join(src_folder,f'{video}.mp4')
    dlc_filepath = os.path.join(pq_folder,f'{video}.parquet')
    output_folder_video = os.path.join(dest_folder,video)
    os.makedirs(output_folder_video, exist_ok = True)

    for roi in ROIs:
        event_df_all = pd.read_csv(os.path.join(events_folder,f'{roi}_events.csv'))
        event_df = event_df_all[event_df_all['video']==video]
        event_df_clips = event_df[(event_df['start_frame']>=10000) & (event_df['end_frame']<=60000)]
        event_df_clips[['video','mouse','start_frame','end_frame','duration']].to_excel(os.path.join(dest_folder,f'validation_{roi}_{video}.xlsx'))

        output_folder = os.path.join(output_folder_video,roi)
        os.makedirs(output_folder, exist_ok = True)
        export_clips_single(events_df = event_df_clips, video_path = video_filepath, path_dlc = dlc_filepath, video_export_path = output_folder)


# %%
src_folder = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\July_2026\cropped_videos\male_P42"
pq_folder=r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\July_2026\parquet\male_P42\baseline"
dest_folder = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\July_2026\validation\ROI"
video_names = ['2026-07-31_18-45-09_crop_5','2026-07-31_18-45-09_crop_6','2026-07-31_18-45-09_crop_7','2026-07-31_18-45-09_crop_8',
               '2026-08-02_06-45-41_crop_1','2026-08-02_06-45-41_crop_2','2026-08-02_06-45-41_crop_3','2026-08-02_06-45-41_crop_4',
               '2026-08-01_02-45-21_crop_1','2026-08-01_02-45-21_crop_2','2026-08-01_02-45-21_crop_3','2026-08-01_02-45-21_crop_4']

ROIs = ['s_wall','ramp1','ramp2','woodstick','feeder_prox', 'feeder_dist','water_prox','water_dist']
events_folder = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\July_2026\ROIs\male_P42\baseline"

for video in video_names:
    video_filepath = os.path.join(src_folder,f'{video}.mp4')
    dlc_filepath = os.path.join(pq_folder,f'{video}.parquet')
    output_folder_video = os.path.join(dest_folder,video)
    os.makedirs(output_folder_video, exist_ok = True)

    for roi in ROIs:
        event_df_all = pd.read_csv(os.path.join(events_folder,f'{roi}_events.csv'))
        event_df = event_df_all[event_df_all['video']==video]
        event_df_clips = event_df[(event_df['start_frame']>=10000) & (event_df['end_frame']<=60000)]
        event_df_clips[['video','mouse','start_frame','end_frame','duration']].to_excel(os.path.join(dest_folder,f'validation_{roi}_{video}.xlsx'))

        output_folder = os.path.join(output_folder_video,roi)
        os.makedirs(output_folder, exist_ok = True)
        export_clips_single(events_df = event_df_clips, video_path = video_filepath, path_dlc = dlc_filepath, video_export_path = output_folder)


# %%
