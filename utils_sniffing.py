#%%
import os
import pandas as pd
import cv2
import numpy as np
import math
from scipy.spatial import distance
from shapely.geometry import Polygon
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist



#%%
def get_prox_data(dict_prox,exp='2CD1_CDSD.exp0001.day01.cam01',pair='BG',type='df'):
    '''this function fetches the proximity events from the dictionary
    Parameters
    dict_prox: dictionary with proximity events
    exp: experiment names, same as the key in the dictionary
    pair: mouse pair, same as the key in the dictionary
    type: export type; 
    - 'df': a dataframe with start, stop columns
    - 'list': a list of all the positive frames'''
    if type =='df':
        df = dict_prox[exp][pair][0]
        result = df.rename(columns={0:'Start',1:'Stop'})
    if type =='list':
        result = dict_prox[exp][pair][1]
    return result
#%% 
def map_columns(column_name):
    '''Get mouse, body part, coord information from dlc parquent file which does not have column names
    For renaming columns'''
    # extract index from column name
    index_str = column_name.split('.')[-1]
    index = 0 if not index_str.isdigit() else int(index_str)
    # get mouse, body part, coord type
    mouse =['Red','Blue','Green','Yellow'][index // 24]
    body_part = ['1_earL','2_earR','3_nose','4_centre','5_centreL','6_centreR','7_tailBase','8_tailTip'][(index % 24) // 3]
    #body_part = (index % 24) // 3 + 1
    coord = ['x', 'y', 'likelihood'][(index % 24) % 3]
    return f'{mouse}_{body_part}_{coord}'
#%% Define a function to subset dataframe by mouse, bp or coord type
def subset_df(df,mouse=None,bp=None,coord=None,index_list=None):
    '''Subset the dlc dataframe for any given mouse pair, bp, coordinate types ('x','y','likelihood'), or frame indexes
    All conditions should be given as lists'''
    mask = pd.Series(True, index=df.columns)    # Initialize a boolean mask for filtering
    if mouse is not None:
        mouse_conditions = [df.columns.str.startswith(f'{m}_') for m in mouse]
        #mask &= pd.concat(mouse_conditions,axis=1).any(axis=1)
        mask &= pd.concat([pd.Series(cond, index=df.columns) for cond in mouse_conditions], axis=1).any(axis=1)
    if bp is not None:
        bp_conditions = [df.columns.str.contains(f"_{b}_") for b in bp]
        #mask &= pd.concat(bp_conditions,axis=1).any(axis=1)
        mask &= pd.concat([pd.Series(cond, index=df.columns) for cond in bp_conditions], axis=1).any(axis=1)
    if coord is not None:
        coord_conditions = [df.columns.str.endswith(f'_{c}') for c in coord]
        #mask &= pd.concat(coord_conditions,axis=1).any(axis=1)
        mask &= pd.concat([pd.Series(cond, index=df.columns) for cond in coord_conditions], axis=1).any(axis=1)

    subset = df.loc[:,mask]
    if index_list is not None:
        subset = subset.loc[index_list]
    return subset

# %%
def get_integers_between_start_stop(dataframe):
    '''This function converts an event dataframe to list
    e.g., a dataframe with start, stop frames for events -> list of all positive frames'''
    result = []
    for _, row in dataframe.iterrows():
        start, stop = row['Start'], row['Stop']
        integers_range = list(range(start, stop + 1))
        result.extend(integers_range)
    return result

#%%
def get_distance_matrices2(dataframe, mouse1, mouse2,body_parts = ['1_earL','2_earR','3_nose','4_centre','5_centreL','6_centreR','7_tailBase']):
    # Initialize a dict to store the matrices
    distance_matrices = []
    #index_array = []
    for index, row in dataframe.iterrows():
        coords_mouse1 = [(row[f'{mouse1}_{bp}_x'],row[f'{mouse1}_{bp}_y']) for bp in body_parts]
        coords_mouse2 = [(row[f'{mouse2}_{bp}_x'],row[f'{mouse2}_{bp}_y']) for bp in body_parts]
        dist_matrix = cdist(coords_mouse1,coords_mouse2,'euclidean')
        distance_matrices.append(dist_matrix)
        #index_array.append(index)
    return distance_matrices
#%%
def get_likelihood_matrices2(dataframe, mouse1, mouse2,body_parts = ['1_earL','2_earR','3_nose','4_centre','5_centreL','6_centreR','7_tailBase']):
    # Initialize a dict to store the matrices
    likelihood_matrices = []
    #index_array = []
    for index, row in dataframe.iterrows():
        l_mouse1 = np.array([row[f'{mouse1}_{bp}_likelihood'] for bp in body_parts])
        l_mouse2 = np.array([row[f'{mouse2}_{bp}_likelihood'] for bp in body_parts])
        l_matrix = np.minimum(l_mouse1[:, np.newaxis], l_mouse2)
        likelihood_matrices.append(l_matrix)
        #index_array.append(index)
    return likelihood_matrices



#%%
def sniffings(contact_matrix,likelihood_matrix,weight_matrix=[1,0.8,1]):
    # Leave out uncertain tracking
    #contact_matrix = np.array(contact_matrix)
    #likelihood_matrix = np.array(likelihood_matrix)
    contact_matrix[likelihood_matrix !=1 ] = np.nan

    range_dist = np.nanmax(contact_matrix) - np.nanmin(contact_matrix)
    # number of invalid values for all BPs
    na_head2head=np.isnan([i for j in contact_matrix[0:3,0:3] for i in j])
    na_anogenital1=np.isnan([i for j in contact_matrix[6:7,0:3] for i in j])
    na_anogenital2=np.isnan([i for j in contact_matrix[0:3,6:7] for i in j])
    na_head2side1=np.isnan([i for j in contact_matrix[3:6,0:3] for i in j])
    na_head2side2=np.isnan([i for j in contact_matrix[0:3,3:6] for i in j])
    # Distance with only nose
    nose2head1=np.nanmedian([i for j in contact_matrix[0:3,2:3] for i in j])
    nose2head2=np.nanmedian([i for j in contact_matrix[2:3,0:3] for i in j])
    nose2side1=np.nanmedian([i for j in contact_matrix[3:6,2:3] for i in j])
    nose2side2=np.nanmedian([i for j in contact_matrix[2:3,3:6] for i in j])
    nose2tail1=np.nanmedian([i for j in contact_matrix[6:7,2:3] for i in j])
    nose2tail2=np.nanmedian([i for j in contact_matrix[2:3,6:7] for i in j])
    sniffings_nose = ['head2head','head2head','head2side1','head2side2','anogenital1','anogenital2']#[1,1,2,3,4,5]
    nose = [nose2head1*weight_matrix[0],nose2head2*weight_matrix[0],nose2side1*weight_matrix[1],nose2side2*weight_matrix[1],nose2tail1*weight_matrix[2],nose2tail2*weight_matrix[2]]
    
    #weights [head,side,tail]
    # Distance with head
    head2head=np.nanmedian([i for j in contact_matrix[0:3,0:3] for i in j])
    anogenital1=np.nanmedian([i for j in contact_matrix[6:7,0:3] for i in j])
    anogenital2=np.nanmedian([i for j in contact_matrix[0:3,6:7] for i in j])
    head2side1=np.nanmedian([i for j in contact_matrix[3:6,0:3] for i in j])
    head2side2=np.nanmedian([i for j in contact_matrix[0:3,3:6] for i in j])
    sniffings_head=['head2head','anogenital1','anogenital2','head2side1','head2side2']#[1,4,5,2,3]
    head_unweighted = [head2head,anogenital1,anogenital2,head2side1,head2side2]
    head =[head2head*weight_matrix[0],anogenital1*weight_matrix[2],anogenital2*weight_matrix[2],head2side1*weight_matrix[1],head2side2*weight_matrix[1]]
    # Assign sniffing type
    #if not np.all(np.isnan(head)) and np.nanargmin(head_unweighted)!=0 and np.sum(np.array(head_unweighted)>50) >3:
        #return None
    if not np.all(np.isnan(nose)) and np.nanmin(nose)<=0.3*range_dist:
        return sniffings_nose[np.nanargmin(nose)] 
    elif not np.any(np.isnan(head)):
        if np.nanmin(head) >= 0.4*range_dist:
            return None
        else:
            return sniffings_head[np.nanargmin(head)]
    elif not np.all(np.isnan(head)):
        if np.sum(na_head2head)<=6 and np.nanmin(head)<0.5*range_dist:
            return sniffings_head[np.nanargmin(head)]
        if np.nanmin(head_unweighted)<0.5*range_dist:
            return sniffings_head[np.nanargmin(head_unweighted)]
        #if (np.isnan(anogenital1) or np.isnan(anogenital1)) and np.nanmin(head)<=50:
            #return sniffings_head[np.nanargmin(head)]
    else:
        return None


# %% Connect events
#%% Connect events for each type separately
def find_bouts(list_frames,threshold_length,threshold_gap):
    '''Define bouts based on consecutive frames distance being < treshold_gap and bout length > treshold_time
    returns df of start and end frame for validation'''
    start_frame=[]
    end_frame=[]
    i=0
    count=1
    
    if len(list_frames)==0:
        return pd.DataFrame([], columns=[0, 1])

    while i+count < len(list_frames):
        if list_frames[i] not in start_frame:
            start_frame.append(list_frames[i]) 

        if list_frames[i+count]-list_frames[i+count-1]> threshold_gap :
            end_frame.append(list_frames[i+count-1])
            i+=count
            count=1

        else:
            i=i
            count+=1
    
    if count==1:
        start_frame.append(list_frames[len(list_frames)-1])
        end_frame.append(list_frames[len(list_frames)-1])
    else:
        end_frame.append(list_frames[len(list_frames)-1])

    frames=[[i,j] for i,j in zip(start_frame,end_frame) if j-i>=threshold_length ]
    if not frames:  # Check if frames list is empty after loop
        return pd.DataFrame([], columns=[0, 1])  # Return empty DataFrame
    
    frames=pd.DataFrame(frames)
    return(frames) 
def isin_list(frame,frame_list):
    return 1 if frame in frame_list else 0
#%%
def add_prediction(df,sniffing_type,thres_len=3,thres_gap=6):
    '''Add columns for each sniffing type to the data frame, where value is 1 if the frame is predicted as the corresponding type and 0 if not.
    '''
    frames = df[df['Prediction']==sniffing_type]['Index'].tolist()
    bouts_df = find_bouts(list_frames=frames,threshold_length= thres_len,threshold_gap=thres_gap)
    bouts_df = bouts_df.rename(columns={0:'Start',1:'Stop'})
    predicted_frames = get_integers_between_start_stop(bouts_df)
    df[f'{sniffing_type}_event'] = df['Index'].apply(lambda x :isin_list(x,predicted_frames))


#%%
def get_prediction_dicts(df):
    '''go through the predictions for each type
    if the frame has no prediction -> None
    if the frame has only one type prediction -> assign that type
    if the frame has multiple predictions -> assign 'uncertain', and
    make an entry in the multiple_prediction dictionary with the frame number as key and all predictions a list'''
    df['count'] = df['head2head_event'] + df['head2side1_event'] + df['head2side2_event'] + df['anogenital1_event'] + df['anogenital2_event']
    final_prediction = {}
    multiple_prediction = {}
    for index, row in df.iterrows():
        if row['count'] == 0 or row['count']>=3:
            final_prediction[row['Index']]= None
        elif row['count'] == 1:
            for col in df.columns[4:]:
                if row[col]==1:
                    final_prediction[row['Index']] = col
                    break
        elif (row['head2head_event']==1) & ((row['anogenital1_event']==1) | (row['anogenital2_event']==1)):
            final_prediction[row['Index']] = None
        else:
            final_prediction[row['Index']] = 'uncertain'
            multiple_prediction[row['Index']] = []
            for col in df.columns[4:9]:
                if row[col]==1:
                    multiple_prediction[row['Index']].append(col)
    return final_prediction,multiple_prediction

#%%
def modify_predictions2(predictions):
    '''Modify the frames with uncertain predictions. 
    For consecutive frames with 'uncertain', assign the first half to the previous frame with a certain prediction, and the last half to the next frame with certain prediction.'''
    modified_predictions = predictions.copy()   # Create a copy to avoid modifying original
    #current_prediction = None
    uncertain_start = None
    uncertain_count = 0

    for frame_number, prediction in predictions.items():
        if prediction == 'uncertain':
            if not uncertain_start:
                uncertain_start = frame_number
            uncertain_count += 1
        else:
            if uncertain_count > 0:
                if uncertain_count > 10:
                    for i in range(uncertain_start,uncertain_start+uncertain_count):
                        if i in predictions:
                            modified_predictions[i] = None
                else:
                    p_prev = predictions[uncertain_start-1] if uncertain_start-1 in predictions else None
                    p_next = prediction if frame_number-(uncertain_start+uncertain_count)<=2 else None
                    mid_point = uncertain_start + (uncertain_count // 2)
                    for i in range(uncertain_start,mid_point):
                        if i in predictions:
                            modified_predictions[i] = p_prev
                    for i in range(mid_point,uncertain_start+uncertain_count):
                        if i in predictions:
                            modified_predictions[i] = p_next 
            uncertain_count = 0
            uncertain_start = None
    # check if ends with uncertain 
    if uncertain_count > 0:
        if uncertain_count > 25:
            for i in range(uncertain_start,uncertain_start+uncertain_count):
                if i in predictions:
                    modified_predictions[i] = None
        else:
            p_prev = predictions[uncertain_start-1] if uncertain_start-1 in predictions else None
            p_next = None
            mid_point = uncertain_start + (uncertain_count // 2)
            for i in range(uncertain_start,mid_point):
                if i in predictions:
                    modified_predictions[i] = p_prev
            for i in range(mid_point,uncertain_start+uncertain_count):
                if i in predictions:
                    modified_predictions[i] = p_next 
    return modified_predictions
 
                
#%%
def get_events(predictions_dict):
    '''Reconnect event after modified prediction
    filter on length? '''
    events = []
    current_event = None
    start_frame = None

    for frame_number, prediction in sorted(predictions_dict.items()):
        #print(frame_number)
        if frame_number+1 not in predictions_dict:
            #print("1")
            #Gap in frame number
            if current_event is not None:
                events.append({"Type":current_event,"Start":start_frame,"Stop":frame_number})
                #print("2")
            current_event = None
            start_frame = frame_number
            continue
        if prediction != current_event:
            #print("3")
            #Start new event
            if current_event is not None:
                events.append({"Type":current_event,"Start":start_frame,"Stop":frame_number - 1})
                #print("4")
            current_event = prediction
            start_frame = frame_number
        else:
            #print("5")
            #Continue same event
            pass
    if current_event is not None:
        last_frame = max(predictions_dict.keys())
        events.append({"Type":current_event,"Start":start_frame,"Stop":last_frame})
    if not events:
        return pd.DataFrame(columns=["Type", "Start", "Stop"])
    return pd.DataFrame(events)


#%%
def read_data(path_proximity, path_dlc, exp_name, pair_name):
    '''Read in the proximity file and DLC file, and return a data frame containing the proximity frames of the the pair of interest, 
    and the calculated distance and likelihood matrices.

    path_proximity: path to the pickle file
    path_dlc: path to DLC output
    exp_name: video name without any extensions e.g.,'2CD1_CDSD.exp0001.day01.cam01'
    pair_name: mouse pairs, e.g., 'BG' '''
    dict_prox = pd.read_pickle(path_proximity)
    list_prox_frames = get_prox_data(dict_prox=dict_prox,exp=exp_name,pair=pair_name,type='list')
    df_dlc = pd.read_parquet(path_dlc)
    df_dlc.rename(columns=map_columns,inplace=True)
    
    mouse_dict = {'B':'Blue','Y':'Yellow','R':'Red','G':'Green'}
    mice_list = [mouse_dict[i] for i in [*pair_name]]
    df_dlc_prox = subset_df(df_dlc,
                            mouse=mice_list,
                            index_list=list_prox_frames)
    df_data = pd.DataFrame({'Index':df_dlc_prox.index,
                           'Dist': get_distance_matrices2(df_dlc_prox,mouse1=mice_list[0],mouse2=mice_list[1]),
                           'LH': get_likelihood_matrices2(df_dlc_prox,mouse1=mice_list[0],mouse2=mice_list[1]),
                           })
    df_data['Pair'] = pair_name
    return df_data
#%%
def predict_events(path_proximity, path_dlc, exp_name, pair_name):
    # Read data
    df_data = read_data(path_proximity=path_proximity, path_dlc=path_dlc,exp_name=exp_name,pair_name=pair_name)
    print(len(df_data))
    if len(df_data)==0:
        return pd.DataFrame(columns=['Type', 'Start', 'Stop', 'Length', 'Pair', 'Video', 'Exp', 'Day', 'Video_split', 'Run'])
    # Predict each frame
    df_data['Prediction'] = df_data.apply(lambda row: sniffings(contact_matrix=row['Dist'],likelihood_matrix=row['LH'],weight_matrix=[1,0.8,1]),axis=1)
    # connect events for each type
    for sniffing in ['head2head','head2side1','head2side2']:
        add_prediction(df=df_data,sniffing_type=sniffing,thres_len=3,thres_gap=2)
    for sniffing in ['anogenital1','anogenital2']:
        add_prediction(df=df_data,sniffing_type=sniffing,thres_len=2,thres_gap=1)
    pred_dict, mul_pred = get_prediction_dicts(df=df_data)
    modified_predictions = modify_predictions2(pred_dict)
    events_df = get_events(modified_predictions)
    events_df['Length'] = events_df['Stop'] - events_df['Start']
    events_df['Pair'] = pair_name
    events_df['Video'] = exp_name   #'1_ELS_Safit_CD1.exp0002.day01.cam02','1CD1_CDSD.exp0003.day03.cam06',2_ELS_Safit_CD1_run2.exp0002.day06.cam02
    events_df['Exp'] = exp_name.split('.')[1]
    events_df['Day'] = exp_name.split('.')[2]
    events_df['Video_split'] = exp_name[0]
    events_df['Run'] = 2 if 'run2' in exp_name else 1   # Only applicable for ELS videos
    return events_df

def find_events_all_pairs(info):
    proximity_dict_path, dlc_path, video_name = info
    pair_dfs = []
    for pair in ['BG', 'BY', 'GY', 'RB', 'RG', 'RY']:
        pair_df = predict_events(path_proximity=proximity_dict_path,
                                 path_dlc=dlc_path,
                                 exp_name=video_name,
                                 pair_name=pair)
        pair_dfs.append(pair_df)
    result_df = pd.concat(pair_dfs)

    return (result_df,video_name)


# %% Export video clips
# Adapted script from Daniil
def export_clips(events_df, video_path,path_dlc, video_export_path, fps=25, buffer_frames=0):
    df_dlc = pd.read_parquet(path_dlc)
    df_dlc.rename(columns=map_columns,inplace=True)
    # Get a list of all video files in the folder (assuming .mp4 files, but you can adjust as needed)
    #all_video_files = [f for f in os.listdir(video_folder_path) if f.endswith('.mp4')]
    if not os.path.exists(video_export_path):
        os.makedirs(video_export_path)
    # Make clips for each event
    for index, row in events_df.iterrows():
        mouse_pair = row['Pair']
        start_frame = max(0, row['Start'] - buffer_frames)
        end_frame = row['Stop'] + buffer_frames
        #label = 'TP' if row['Is_annotated']==1 else 'FP'
        sniffing_type = row['Type'].split('_')[0]

        bodyparts = ['1_earL','2_earR','3_nose','4_centre','5_centreL','6_centreR','7_tailBase','8_tailTip']
        mouse_dict = {'B':'Blue','Y':'Yellow','R':'Red','G':'Green'}
        individuals = [mouse_dict[i] for i in [*mouse_pair]]

        #clip_folder = f'clips_{mouse_pair}_{label}'
        clip_folder = f'clips_{mouse_pair}'
        clip_path = os.path.join(video_export_path,clip_folder)
        if not os.path.exists(clip_path):
            os.makedirs(clip_path)
        #clip_name = f'{index}_{mouse_pair}_{start_frame}_{end_frame}_{sniffing_type}_{label}.mp4'
        clip_name = f'{index}_{mouse_pair}_{start_frame}_{end_frame}_{sniffing_type}.mp4'
        clip_filepath = os.path.join(clip_path,clip_name)
        print(clip_filepath)

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

            for mouse in individuals:
                for part in bodyparts:
                    x = row[f'{mouse}_{part}_x']
                    y = row[f'{mouse}_{part}_y']

                    if (not math.isnan(x)) & (not math.isnan(y)):
                        if mouse == individuals[0]:
                            cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
                        else:
                            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)
            
            current_frame += 1
            out.write(frame)
        cap.release()
        out.release()