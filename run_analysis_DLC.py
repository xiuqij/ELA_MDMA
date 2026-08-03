#%% Import DLC (activate DLC environment first)
import deeplabcut

#%% Define paths for model and videos
config_path=r"L:\Lopez Laboratory - NEURO\SB_Pilot\Sexy16\DLC\Sexy16Pilot-Sere-2025-03-10\config.yaml"  
videos_path=r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\July_2026\cropped_videos\female_P35"

#%% analyze videos
#output of this step is [videoname]DLC_resnet50_Sexy16PilotMar10shuffle9_180000_full.pickle and [videoname]DLC_resnet50_Sexy16PilotMar10shuffle9_180000_meta.pickle
deeplabcut.analyze_videos(config_path,videos_path,auto_track=False,identity_only=True,videotype='.mp4',shuffle=9)

 #%% convert detections to tracklets
#output of this step is [videoname]DLC_resnet50_Sexy16PilotMar10shuffle9_180000_assemblies.pickle and [videoname]DLC_resnet50_Sexy16PilotMar10shuffle9_180000_el.pickle 
deeplabcut.convert_detections2tracklets(config_path,videos_path,videotype='mp4',track_method='ellipse',identity_only=True,shuffle=9)

#%% stitch tracklets
#output of this step is [videoname]DLC_resnet50_Sexy16PilotMar10shuffle9_180000_el.h5
deeplabcut.stitch_tracklets(config_path,videos_path,videotype='.mp4',split_tracklets=False,track_method='ellipse',shuffle=9)

#%% filter predictions
#output of this step is [videoname]DLC_resnet50_Sexy16PilotMar10shuffle9_180000_el_filtered.h5 and [videoname]DLC_resnet50_Sexy16PilotMar10shuffle9_180000_el_filtered.csv
deeplabcut.filterpredictions(config_path,videos_path,track_method='ellipse',shuffle=9)

#%% [for validation only] make video clips for validation
deeplabcut.create_labeled_video(config_path,videos_path,filtered=True,displayedbodyparts=['1_earL','2_earR','3_nose','4_centre','5_centreL','6_centreR','7_tailBase','8_tailTip'],draw_skeleton=True,color_by='individual',track_method='ellipse',shuffle=9)
# %%
# 