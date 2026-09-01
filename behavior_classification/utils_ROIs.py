#%%
import pandas as pd
import numpy as np
#import cv2
from shapely.geometry import Polygon, Point
import geopandas as gpd
import pickle
import os
import glob
import re
import multiprocessing
from multiprocessing import Pool, set_start_method, pool
import time

#%%
class ROIS(object):
    def __init__(self, arena_dict, roi_names,video ):
        self.dict_roi = {}
        self.dict_roi_vertices = {}
        self.dict_roi_centroids = {}
        self.roi_names = roi_names
        self.poly_vertices =[]

        for index, roi in arena_dict.items():  
            self.poly_vertices = roi
            polygon_shape = Polygon(self.poly_vertices)
            #roi_name = roi_names[index]
            roi_name=index
     
            self.dict_roi[roi_name] = polygon_shape
            self.dict_roi_vertices[roi_name] = self.poly_vertices
            centroid = polygon_shape.centroid
            center_x = centroid.x
            center_y = centroid.y
            center = np.asarray([center_x, center_y])
            self.dict_roi_centroids[roi_name] = center

    def visualize_polygon(self, roi_name, color=(0, 255, 0)):
        width = 1200
        height = 1200
        blank_image = np.zeros((height, width, 3), dtype=np.uint8)

        poly_vertices = self.dict_roi_vertices[roi_name]

        for v in range(1, len(poly_vertices)):
            cv2.line(blank_image, poly_vertices[v], poly_vertices[v - 1], color, 2)

        cv2.line(blank_image, poly_vertices[0], poly_vertices[-1], color, 2)

        cv2.imshow('ROI', blank_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
#%%
def detect_bodypart_in_ROI(df,mouse, bodypart,polygon_gdf, polygon):
    bodypart_df = pd.DataFrame(index=df.index)
    bodypart_df['point_in_roi'] = False
    bodypart_df['likelihood'] = 0
    df = df[~df[(mouse,bodypart,'x')].isna()]   #Remove NaN points
    try:
        # Convert the DataFrame to a GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[( mouse, bodypart, 'x')],
                                                               df[( mouse, bodypart, 'y')]))
        # Create a spatial index for the points
        sindex = gdf.sindex
        # Find the bounding box of the polygon
        bbox = polygon_gdf.geometry.bounds.iloc[0]
        # Get the indices of the points that have a bounding box intersection with the polygon
        possible_matches_index = list(sindex.intersection(bbox))
        # Filter the points using the possible matches indices
        possible_matches = gdf.iloc[possible_matches_index]
        # Now do the precise check to see which of the possible matches actually lie within the polygon
        precise_matches = possible_matches[possible_matches.intersects(polygon)]
        # Create a new column in the original DataFrame to indicate if each point lies within the polygon
        bodypart_df.loc[precise_matches.index, 'point_in_roi'] = True
        bodypart_df['likelihood'] = df[( mouse, bodypart, 'likelihood')]
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return bodypart_df

#%%
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


#%%
def convert_to_df(events,mouse,video):
    '''Create the result data frame with necessary information
    ADAPTED FOR 4-HOUR VIDEOS'''

    events_df = pd.DataFrame(events, columns=['start_frame', 'end_frame'])
    events_df['duration'] = events_df['end_frame'] - events_df['start_frame'] + 1
    events_df['mouse'] = mouse
    
    events_df['video'] = video  #2024-12-17_16-44-02_crop_green
    events_df['box']=video.split('_')[3]
    events_df['date']=video.split('_')[0]
    events_df['timestamp']= video.split('_')[1]

    #events_df['phase']= 'active' if timebin in [19,20,21,22,23,0,1,2,3,4,5,6] else 'inactive'

    # Rearrange the columns
    events_df = events_df[['video','mouse','start_frame', 'end_frame', 'duration','box','date','timestamp']]

    return events_df

#%%
def process_file(file_info):
    '''New function that combines all ROIs
    For ROIs in rois_headbps, the interaction is defined by the head body parts
    For ROIs in rois_allbps, the interaction is defined by all body parts.'''
    # Define column names and body parts
    rois_headbps = ['feeder_prox', 'feeder_dist','water_prox','water_dist']
    rois_allbps = ['s_wall','above_nest','ramp1','ramp2','non_wall','woodstick']
    individuals=['red','blue','green','yellow']
    bps=['1_earL','2_earR','3_nose','4_center','5_centerL','6_centerR','7_tailBase','8_tailTip']
    coords=['x','y','likelihood']
    col_names1=[]
    col_names2=[]
    col_names3=[]
    for individual in individuals:
        for bp in bps:
            for coord in coords:
                col_names1.append(individual)
                col_names2.append(bp)
                col_names3.append(coord)
    col_names=[col_names1,col_names2,col_names3]

    #arenas_roi = {}
    head_parts = ['1_earL', '3_nose', '2_earR']
    torso_parts =   ['4_center','7_tailBase']

    # Read input 
    parquet_file_path,file_path,arenas_roi = file_info   #unpack the input tuple
    video_key=os.path.basename(os.path.splitext(parquet_file_path)[0])
    
    df = pd.read_parquet(parquet_file_path,engine='pyarrow')
    df.columns = pd.MultiIndex.from_arrays(col_names)

    #scorer = df.columns.get_level_values(0)[0]

    likelihood_threshold = 0.05


    dict_polygon = {}
    dict_polygon_gdf ={}
    for roi in rois_headbps:
        roi_polygon = arenas_roi[video_key].dict_roi[roi]
        roi_polygon_gdf = gpd.GeoDataFrame(pd.DataFrame({'geometry': [roi_polygon]}))
        dict_polygon[roi] = roi_polygon
        dict_polygon_gdf[roi] =  roi_polygon_gdf


    all_events = {}

    for roi in rois_headbps:
        all_events[roi] = []
    for mouse in individuals:
        #print("Analysing Mouse   " + mouse)
        features_df =  pd.DataFrame(index=df.index)
        features_df['torso_counter'] = 0
        for roi in rois_headbps:
            features_df[f'head_{roi}_counter'] = 0
            for bodypart in head_parts:
                bodypart_df_head_roi = detect_bodypart_in_ROI(df, mouse, bodypart, dict_polygon_gdf[roi], dict_polygon[roi])
                bodypart_df_head_roi['head_in_roi'] = bodypart_df_head_roi['point_in_roi'] & (bodypart_df_head_roi['likelihood'] >= likelihood_threshold)
                features_df[f'head_{roi}_counter'] += bodypart_df_head_roi['head_in_roi'].astype(int)
            if 'feeder' in roi:
                features_df[f'is_{roi}_frame'] = (features_df[f'head_{roi}_counter'] >= 2)
                roi_events = find_events(features_df[f'is_{roi}_frame'], min_event_duration=25, max_gap=10)
                all_events[roi].append(convert_to_df(roi_events, mouse,video_key))
            if 'water' in roi:
                features_df[f'is_{roi}_frame'] = features_df[f'head_{roi}_counter'] >= 1
                roi_events = find_events(features_df[f'is_{roi}_frame'], min_event_duration=25, max_gap=150)
                all_events[roi].append(convert_to_df(roi_events, mouse,video_key))
                
    for roi in rois_allbps:
        all_events[roi]=[]
    for mouse in individuals:
        #print(f"Analysing mouse: {mouse}\n")
        features_df = pd.DataFrame(index=df.index)
        for roi in rois_allbps:
            roi_polygon = arenas_roi[video_key].dict_roi[roi]
            roi_polygon_gdf = gpd.GeoDataFrame(pd.DataFrame({'geometry':[roi_polygon]}))
            features_df[f'bodypart_in_roi_counter_{roi}']=0
            for bp in bps:
                bodypart_df = detect_bodypart_in_ROI(df,mouse,bp,roi_polygon_gdf,roi_polygon)
                bodypart_df[f'bodypart_in_roi_{roi}'] = bodypart_df['point_in_roi'] & (bodypart_df['likelihood'] >= likelihood_threshold)
                features_df[f'bodypart_in_roi_counter_{roi}'] += bodypart_df[f'bodypart_in_roi_{roi}'].astype(int)
            
            features_df[f'mouse_in_roi_{roi}'] = features_df[f'bodypart_in_roi_counter_{roi}']  > 3

            if roi == 'non_wall':
                features_df['mouse_in_roi_non_wall'] = ~ features_df['mouse_in_roi_non_wall']
                condition = features_df['mouse_in_roi_ramp1'] | features_df['mouse_in_roi_ramp2'] | features_df['mouse_in_roi_above_nest']
                features_df.loc[condition, ['mouse_in_roi_non_wall']] = False
            
            roi_events = find_events(features_df[f'mouse_in_roi_{roi}'], min_event_duration=25, max_gap=50)
            all_events[roi].append(convert_to_df(roi_events,mouse,video_key))
    
    return all_events