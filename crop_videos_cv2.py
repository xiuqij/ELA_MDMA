
# opencv should work, but seems to create larger videos
#%%
import cv2
import os
import pickle
#%% === CONFIGURATION ===
input_folder = "path/to/input/folder"
output_folder = "path/to/output/folder"
saved_coords_path = ''
with open(saved_coords_path,'rb') as f:
    crop_coords = pickle.load(f)
all_folders = os.listdir(input_folder)
# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
#%% function, crop one folder
def crop_videos_in_folder(src_path, dest_path, crop_regions, sb_order = ['red', 'pink', 'orange', 'green']):
    for filename in os.listdir(src_path):
        if filename.endswith('.mp4'):
            video_path = os.path.join(src_path,filename)
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Failed to open {filename}")
                continue
            # Video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # Setup video writers for 4 crops
            basename = os.path.splitext(filename)[0]
            writers = []

            for i, (x, y, w, h) in enumerate(crop_regions):
                sb_code = sb_order[i]
                out_path = os.path.join(output_folder, f"{basename}_crop_{sb_code}.mp4")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                writers.append(writer)

            print(f"Processing {filename}...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                for i, (x, y, w, h) in enumerate(crop_regions):
                    crop = frame[y:y+h, x:x+w]
                    writers[i].write(crop)

            cap.release()
            for writer in writers:
                writer.release()

            print(f"Finished processing {filename}")

    print("All videos processed.")

#%% process all
for folder_name in all_folders:
    src_folder_path = os.path.join(input_folder,folder_name)
    dest_folder_path = os.path.join(output_folder,folder_name)
    crop_coords_folder = crop_coords[folder_name]
    sb_order = ['red', 'pink', 'orange', 'green']
    crop_videos_in_folder(src_path=src_folder_path,
                          dest_path=dest_folder_path,
                          crop_regions=crop_coords_folder,
                          sb_order=sb_order)
