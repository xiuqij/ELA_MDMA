#%% Import packages
import ffmpeg
import os
import pickle
#%% Functions for cropping and file processing
def crop_and_save_ffmpeg(input_path, output_paths, crop_coords):
    for i, (x1, y1, x2, y2) in enumerate(crop_coords):
        width = x2 - x1
        height = y2 - y1
        output = output_paths[i]

        print(f"Cropping area {i+1} -> {output} ...")
        try:
            (
                ffmpeg
                .input(input_path)
                .crop(x1, y1, width, height)
                .output(output, vcodec='libx264', crf=23, preset='fast', pix_fmt='yuv420p')
                .overwrite_output()
                .run(quiet=True)
            )
            print(f"Saved: {os.path.basename(output)}")
        except ffmpeg.Error as e:
            print(f"Error processing {output}: {e.stderr.decode() if e.stderr else e}")


def process_single_video_ffmpeg(video_file, input_folder, output_folder, crop_coords, sb_labels):
    input_path = os.path.join(input_folder, video_file)
    base, _ = os.path.splitext(video_file)
    output_paths = [
        os.path.join(output_folder, f"{base}_crop_{label}.mp4")
        for label in sb_labels
    ]
    print(f"\nProcessing: {video_file}")
    crop_and_save_ffmpeg(input_path, output_paths, crop_coords)


def get_video_files(folder):
    '''This function returns a list of all .mp4 files in a given folder'''
    return [f for f in os.listdir(folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]


def process_all_videos_ffmpeg(input_folder, output_folder, crop_coords, sb_labels):
    video_files = get_video_files(input_folder)

    if not video_files:
        print("No videos found.")
        return

    print(f"Found {len(video_files)} video(s) in {input_folder}")
    os.makedirs(output_folder, exist_ok=True)

    for vf in video_files:
        process_single_video_ffmpeg(vf, 
        input_folder=input_folder,output_folder=output_folder,crop_coords=crop_coords,sb_labels=sb_labels)
    
    print("\nCompleted.")

#%%
INPUT_PATH = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\July_2026\recordings\male_P42"
OUTPUT_PATH = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\July_2026\cropped_videos\male_P42"
SAVED_COORDS = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\July_2026\crop_coords_july.pickle"
with open(SAVED_COORDS,'rb') as f:
    dict_coords = pickle.load(f)
folders = ['K6_242','K6_243']

os.makedirs(OUTPUT_PATH,exist_ok=True)
for folder_name in folders:
    src_folder_path = os.path.join(INPUT_PATH,folder_name)
    dest_folder_path = OUTPUT_PATH
    crop_coords = dict_coords[folder_name]
    sb_id = ['1','2','3','4'] if folder_name == 'K6_242' else ['6','8','5','7']
    process_all_videos_ffmpeg(input_folder=src_folder_path,output_folder=dest_folder_path,
                              crop_coords=crop_coords,sb_labels=sb_id)

# %%
