import os
from moviepy.editor import VideoFileClip


#function to crop video as desired
def crop_video(input_video_path, output_video_path, crop_area):
    try:
        print(f"Processing video: {input_video_path} with crop area: {crop_area}")
        with VideoFileClip(input_video_path) as video:
            cropped_video = video.crop(x1=crop_area[0], y1=crop_area[1], x2=crop_area[2], y2=crop_area[3])
            cropped_video.write_videofile(output_video_path, codec='libx264', audio=False) #this codec library works. Tried the one for GPU, but it is actually slower
    except Exception as e:
        print(f"Error cropping video {input_video_path}: {e}")

# Function to process a video file with multiple crops
def process_video_file(video_file, input_folder, output_folder, crop_coordinates,colors_SB):
    input_video_path = os.path.join(input_folder, video_file)
    filename, ext = os.path.splitext(video_file)

    # Loop through each set of crop coordinates
    for i, crop_area in enumerate(crop_coordinates):
        output_video_path = os.path.join(output_folder, f"{filename}_crop_{colors_SB[i]}{ext}")
        crop_video(input_video_path, output_video_path, crop_area)

# Function to process videos in a folder
def get_video_files(input_folder):
    return [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov','.mkv'))]