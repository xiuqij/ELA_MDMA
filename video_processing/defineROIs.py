#%%
# Activate venv_easy_roi (python 3.6)
# source venv_easy_roi/bin/activate
from EasyROI import EasyROI
import cv2
import pandas as pd
import os
import pickle
# %%
roi_helper = EasyROI(verbose=True)
# %% Create list of videos to draw

# [change here] folder path 
sample_path= r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\July_2026\samples\samples_ROI"
EXP = 'male_P42'
videos_path = os.path.join(sample_path,EXP)
# check the videos in the folder
videos = [f for f in os.listdir(videos_path)]

videos_to_draw = [os.path.join(videos_path, video) for video in videos]
print(videos_to_draw)
#check if it's the right number of videos
print(len(videos_to_draw))  #8

# %% Create ROI for enrichment and feeding objects
def draw_ROIs(video_path):
    '''draws the ROI for the enrichment objects
    order to draw: s-wall, nest, stick, center (saved in rect_roi); feeder_prox, feeder_dist,
    ramp1 (prox side),ramp2 (saved in poly_roi), water_prox, water_dist'''
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), 'Cannot capture source'
    ret, frame = cap.read()
    rect_roi = roi_helper.draw_rectangle(frame, 6)  # 6 rectangles
    frame_temp_rect = roi_helper.visualize_roi(frame, rect_roi)

    poly_roi= roi_helper.draw_polygon(frame, 4)  # 4 polygons
    frame_temp_poly = roi_helper.visualize_roi(frame, poly_roi)
    return rect_roi,poly_roi

# %% Draw and save ROIs
'''A window will pop up, draw the ROIs in the following order:
    [rectangles] s-wall, nest, woodstick, non-wall (center of the arena), feeder_prox, feeder_dist,
    [polygons] ramp1 (on prox side),ramp2 (the other one), water_prox, water_dist
    Then a new video will pop up, repeat until it's finished for all the videos in the folder
    '''
# Define ROI names
roi_names_rect = ['s_wall','above_nest','woodstick' ,'non_wall','feeder_prox','feeder_dist']
roi_names_poly = ['ramp1', 'ramp2','water_prox','water_dist']
# Define dict to save ROIs for all videos in the folder
roi_dict= {}
# Iterate over the videos 
for video in videos_to_draw:
    # get the video title without extension
    filename = os.path.splitext(os.path.split(video)[1])[0]  #e.g. 2025-07-07_19-01-02_crop_green
    #filename = os.path.os.path.split(video)[1]
    # get the color of the box (use the SB color as key for easier re-use on the same set of videos)
    sb_color = filename.split("_")[-1]
    
    enrichment_rect, enrichment_poly = draw_ROIs(video)
    roi_dict[sb_color]={}
    for index,coord in enrichment_rect['roi'].items():
        tl = (coord['tl_x'],coord['tl_y'])
        tr = (coord['br_x'],coord['tl_y'])
        br = (coord['br_x'],coord['br_y'])
        bl = (coord['tl_x'],coord['br_y'])
        coords = [tl,tr,br,bl]
        roi_dict[sb_color][roi_names_rect[index]] = coords
    for index,coord in enrichment_poly['roi'].items():
        coords = coord['vertices']
        roi_dict[sb_color][roi_names_poly[index]] = coords


#%% Store ROIs to a pickle file for later use
#[change here] ROIs work folder
ROI_folder = 'L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\July_2026\ROIs'
#[change here] Name the ROI file as the video group e.g., roi_ctrl_after_stress2.pickle
roi_file = f'{EXP}.pickle'
# %% Store the ROI dict in a .pickle file
with open(os.path.join(ROI_folder,roi_file),'wb') as f:
    pickle.dump(roi_dict,f)
    f.close()

# %% Sanity check - Read the ROI file and check the keys
roi = pd.read_pickle(os.path.join(ROI_folder,roi_file))
print(len(roi.keys()))   #8
print(roi)

# %%
