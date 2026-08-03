import os
import pandas as pd
import numpy as np
from scipy.spatial import distance
from shapely.geometry import Polygon
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
import re

#more precise with 6 bps, but if you remove some it will be faster
def remove_bps(df):
    rem_tailtip=[[i,i+1,i+2] for i in range(21,48,24)]
    rem_tailtip=[i for j in rem_tailtip for i in j ]
    rem_center=[[i,i+1,i+2] for i in range(9,48,24)] #center body point is removed for areas computations
    rem_center=[i for j in rem_center for i in j ]
    
    indices_to_remove=sorted(rem_center+rem_tailtip)
    df=df.drop(df.columns[indices_to_remove],axis=1) #check if it doesnt fuck up indeces
    return(df)

def calculate_area(df_row,extension_area):
    polygon1=[(df_row.iloc[i],df_row.iloc[i+1]) for i in range(0,(len(df_row)//2)-1,3)] #coordinates [(x,y),(x1,y1),...] for each bp of animal 1
    polygon2=[(df_row.iloc[i],df_row.iloc[i+1]) for i in range(len(df_row)//2,len(df_row),3)]  #coordinates [(x,y),(x1,y1),...] for each bp of animal 2
    polygon1=sorted([(x, y) for x, y in polygon1 if np.isnan(x)==False and np.isnan(y)==False and x!=0 and y!=0 and np.isinf(x)==False and np.isinf(y)==False])
    polygon2= sorted([(x, y) for x, y in polygon2 if np.isnan(x)==False and np.isnan(y)==False and x!=0 and y!=0 and np.isinf(x)==False and np.isinf(y)==False])

    if len(polygon1)>2:
        polygon1=Polygon(polygon1)
    else: return

    if len(polygon2)>2:
        polygon2=Polygon(polygon2)
    else:return

    if not polygon1.is_valid:  #e.g. there are self-intersections
        polygon1 = polygon1.buffer(0)

    if not polygon2.is_valid:
        polygon2 = polygon2.buffer(0)

    area_no_ext1=polygon1
    area_no_ext2=polygon2

    polygon1=polygon1.buffer(extension_area)
    polygon2=polygon2.buffer(extension_area)
    

    return([polygon1,polygon2,area_no_ext1,area_no_ext2])


def interesect_area(polygon1, polygon2,px_to_mm):
    intersection=(polygon1.intersection(polygon2).area)/px_to_mm
    return intersection

def find_bouts(list_frames,treshold_time,treshold_gap):
    start_frame=[]
    end_frame=[]
    i=0
    count=1
    
    if len(list_frames)==0:
        return

    while i+count < len(list_frames):
        if list_frames[i] not in start_frame:
            start_frame.append(list_frames[i]) 

        if list_frames[i+count]-list_frames[i+count-1]> treshold_gap:
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

    frames=[[i,j] for i,j in zip(start_frame,end_frame) if j-i>=treshold_time ]
    frames=pd.DataFrame(frames)
    
    return(frames) 


def all_frames_from_bouts(df):
    try:
        frames=[]
        for row in range(len(df.iloc[:,0])):
            row_frames=[i for i in range(df.iloc[row,0],df.iloc[row,1]+1)]
            frames.append(row_frames)

    except: ValueError
    frames=[i for j in frames for i in j]
   
    return(frames)


def exclude_other_behavior(df_behav,df_proximity,treshold_time,treshold_gap):
    updated_proximity={}
    for file in list(df_proximity.keys()):
        updated_proximity[file]={}
        for pair in list(df_proximity[file].keys()):
            updated_proximity[file][pair]={}
            frames_to_exclude_an1=all_frames_from_bouts(df_behav[file][pair][0])
            frames_to_exclude_an2=all_frames_from_bouts(df_behav[file][pair][1])
            frames_to_exclude=frames_to_exclude_an1+frames_to_exclude_an2
            frames_to_exclude=set(frames_to_exclude) #should be faster
            
            frames_prox=all_frames_from_bouts(df_proximity[file][pair][0])
            result = [x for x in frames_prox if x not in frames_to_exclude]
            result_bouts=find_bouts(result,treshold_time,treshold_gap)
            updated_proximity[file][pair][0]=result_bouts
            updated_proximity[file][pair][1]=result

    return(updated_proximity)


def convert_to_df(events,pair,video):
# Create a DataFrame for the events
    pattern_days=r'day\d{2}'
    pattern_experiment=r'exp\d{4}'

    events_df=events
    if events_df is not None:
        events_df.columns=['start_frame','end_frame']
        events_df['duration'] = events_df['end_frame'] - events_df['start_frame'] + 1
        events_df['pair'] = pair
    
        events_df['video'] = video
        events_df['manual'] = 0
        events_df['experiment']=re.findall(pattern_experiment,video)[0][-2:]
        events_df['day']=re.findall(pattern_days,video)[0][-2:]
        events_df['videosplit']=video[0]
        # Rearrange the columns
        events_df = events_df[['video','pair', 'duration','experiment','day','videosplit','start_frame', 'end_frame']]

    else:
        events_df=pd.DataFrame([[None,None]],columns=['start_frame','end_frame'])
        events_df['duration'] = None
        events_df['pair'] = pair
        events_df['video'] = video
        events_df['manual'] = 0
        events_df['experiment']=re.findall(pattern_experiment,video)[0][-2:]
        events_df['day']=re.findall(pattern_days,video)[0][-2:]
        events_df['videosplit']=video[0]
        events_df = events_df[['video','pair', 'duration','experiment','day','videosplit','start_frame', 'end_frame']]

    return events_df

def compute_general_stats(df,fps):
    try:
        total_duration=sum(df.loc[:,'duration'])/fps
        n_bouts=len(df.index)
        av_bout_length=total_duration/n_bouts
    except: ValueError; return(0,0,0)
    return(total_duration,n_bouts,av_bout_length)


def processer(info):
    parquet_file,folder_path_parquet,pairs,treshold_time,treshold_gap,extension_area_px,px_to_mm=info
    intersect_dict={}
    bouts_dict={}
    areas_dict={}


    pq_file_path =parquet_file
    df=pd.read_parquet(pq_file_path, engine='pyarrow')
    #df=df.iloc[range(0,90000),:]  #run this line only to compare with 1h-long validations
    file=os.path.basename(pq_file_path)
    file=os.path.splitext(file)[0]
    

    for pair in sorted(list(pairs.keys())):
        cols_pair=pairs[pair]
        df2=df.copy()
        df2=df2.iloc[:,cols_pair]
        df2=remove_bps(df2)
        intersect_dict[pair]={}
        bouts_dict[pair]={}
        areas_dict[pair]={}
        areas_dict[pair][0]={}

        for frame in df2.index:
            df_row=df2.iloc[frame,:]
            areas=calculate_area(df_row,extension_area_px)
            if areas is not None:
                area1=areas[0].area
                area2=areas[1].area
            else: area1=None; area2=None


            if area1 is not None and area2 is not None:

                intersection=interesect_area(areas[0],areas[1],px_to_mm)
                if intersection!=0:
                    intersect_dict[pair][frame]=intersection
                    areas_dict[pair][0][frame]=(areas[2].area,areas[3].area)

        intersections=[j for j in sorted(list(intersect_dict[pair].keys()))]
        areas_list=[j for j in sorted(list(areas_dict[pair][0].keys()))]
        bouts_dict[pair]=[find_bouts(intersections,treshold_time,treshold_gap)]
        #bouts_dict[pair].append(utils.all_frames_from_bouts(bouts_dict[pair][0]))
        areas_dict[pair]=find_bouts(areas_list,treshold_time,treshold_gap)
        #areas_dict[pair][2]=utils.all_frames_from_bouts(areas_dict[pair][1])

    return (bouts_dict,file)


def get_integers_between_start_stop(dataframe):
    '''This function converts an event dataframe to list
    e.g., a dataframe with start, stop frames for events -> list of all positive frames'''
    result = []
    for _, row in dataframe.iterrows():
        start, stop = int(row['start_frame']), int(row['end_frame'])
        integers_range = list(range(start, stop + 1))
        result.extend(integers_range)
    return result

def exclude_chases(df_chase, df_prox, thres_time, thres_gap):
    updated_proximity={}
    for file in list(df_prox.keys()):
        updated_proximity[file]={}
        for pair in list(df_prox[file].keys()):
            updated_proximity[file][pair]={}

            frames_to_exclude = get_integers_between_start_stop(df_chase[(df_chase['video']==f'{pair}_{file}')&(df_chase['pair']==pair)])
            frames_prox=all_frames_from_bouts(df_prox[file][pair][0])

            result = [x for x in frames_prox if x not in frames_to_exclude]
            result_bouts=find_bouts(result,thres_time,thres_gap)
            updated_proximity[file][pair][0]=result_bouts
            updated_proximity[file][pair][1]=result
    return(updated_proximity)