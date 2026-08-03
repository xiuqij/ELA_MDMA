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

# %%
# Create ROI for SBs
def draw_SBs(video_path):
    '''draws the 4 SBs, order: tl, tr, bl, br'''
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), 'Cannot capture source'
    ret, frame = cap.read()
    rect_roi = roi_helper.draw_rectangle(frame, 4)  # 4 rectangles
    frame_temp_rect = roi_helper.visualize_roi(frame, rect_roi)

    return rect_roi

# %% 
# Path of sampled videos
videos_path= "L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\July_2026\samples\samples_for_crop"  

# Make a list of videos to draw
videos_to_draw =[]  #list with the **full path** of the videos 
for video in os.listdir(videos_path):
    videos_to_draw.append(os.path.join(videos_path,video))

print(len(videos_to_draw))  

#%% 
# Define dict to save ROIs for all videos
roi_dict= {}
# Iterate over videos
for video in videos_to_draw:    
    filename = os.path.splitext(os.path.split(video)[1])[0]  #the video names were renamed to the folder where they were sampled from

    sb_rect = draw_SBs(video)
    roi_dict[filename]={}
    for index,coord in sb_rect['roi'].items():
        tl = (coord['tl_x'],coord['tl_y'])
        tr = (coord['br_x'],coord['tl_y'])
        br = (coord['br_x'],coord['br_y'])
        bl = (coord['tl_x'],coord['br_y'])
        coords = [tl,tr,br,bl]
        roi_dict[filename][index] = coords

# %%
print(roi_dict)

# %% Rearrange to fit cropping function
crop_coords = {}
for k,v in roi_dict.items():
    coord_list = []
    for box in [0,1,2,3]:
        x1,y1 = v[box][0]
        x2,y2 = v[box][2]
        coord_list.append((x1,y1,x2,y2))
    crop_coords[k] = coord_list

#%%
print(crop_coords)
#%%
save_path = r'L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\July_2026\crop_coords_july.pickle'
with open(save_path,'wb') as f:
    pickle.dump(crop_coords, f)
# %%
