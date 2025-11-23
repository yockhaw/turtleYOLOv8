import os
import pandas as pd
import time
from functions.modelrunfuncs import *

# Other than saving detection results through videos, which are the bounding boxes, this file reannotates the bounding boxes 
# onto the original frame from which the bounding boxes were detected for result verification later on
# Annotated frames saved in duplicatevideoframesv2

# Paths
# videos
ogvideo_folder = '/Volumes/External SS/cleanvideos'
# Flight record data downloaded from Airdata
flight_folder = '/Users/fooyockhaw/Desktop/allflightdata'
# output of A
metadata_path = '/Users/fooyockhaw/Desktop/video_metadata_categorizedv3.csv'
# output of this file 1
resized_video_folder = '/Users/fooyockhaw/Desktop/resized_videosv2'
# output of this file 2
detection_output_folder = '/Users/fooyockhaw/Desktop/videoresultsv5'
# output of this file 3
summary_csv_path = '/Users/fooyockhaw/Desktop/video_processing_timesv3.csv'
# output of Google Colab Code to train YOLO model
model_path = "/Users/fooyockhaw/Desktop/YOLOv8S.pt"
# output of this file 4
frame_screencapture_folder = '/Users/fooyockhaw/Desktop/duplicatevideoframesv2'

os.makedirs(resized_video_folder, exist_ok=True)
os.makedirs(detection_output_folder, exist_ok=True)
os.makedirs(frame_screencapture_folder, exist_ok=True)

# Load metadata
needs_trim_dict, end_time_dict, start_time_dict, flight_record_dict, flight_time_bounds = load_metadatav2(metadata_path, flight_folder)
metadata_dicts = (needs_trim_dict, end_time_dict, start_time_dict, flight_record_dict, flight_time_bounds)

paths = {
    'og': ogvideo_folder,
    'resized': resized_video_folder,
    'detection': detection_output_folder,
    'screencap': frame_screencapture_folder
}

video_extensions = ('.mp4', '.avi', '.mov')
time_records = []

video_files_to_run = [
    f for f in os.listdir(paths['resized']) 
    if f.lower().endswith(video_extensions)
]

for video_file in video_files_to_run:
    if video_file.lower().endswith(video_extensions):
        # Strip "_640x640" from the filename to match metadata keys
        base_name_clean = video_file.replace("_640x640", "")
        record = process_videov2(base_name_clean, paths, metadata_dicts, model_path, save_frames=True)
        if record:
            time_records.append(record)
    else:
        print(f"{video_file} does not have a valid video extension, skipping.")

# Save summary CSV
if time_records:
    df_summary = pd.DataFrame(time_records)
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Summary saved to {summary_csv_path}")