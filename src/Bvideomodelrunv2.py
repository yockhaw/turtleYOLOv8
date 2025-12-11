import os
import pandas as pd
import time
from functions.modelrunfuncs import *
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=""
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--ogvideo_folder",
        type=str,
        required=True
    )

    parser.add_argument(
        "--flight_folder",
        type=str,
        required=True
    )

    parser.add_argument(
        "--metadata_categorized",
        type=str,
        required=True
    )

    parser.add_argument(
        "--resized_video_folder",
        type=str,
        required=True
    )

    parser.add_argument(
        "--detection_output_folder",
        type=str,
        required=True
    )

    parser.add_argument(
        "--summary_csv_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output_frame_screencapture_folder",
        type=str,
        required=True
    )

    return parser.parse_args()

def main():
    args = parse_arguments()

    # Other than saving detection results through videos, which are the bounding boxes, this file reannotates the bounding boxes 
    # onto the original frame from which the bounding boxes were detected for result verification later on
    # Annotated frames saved in duplicatevideoframesv2

    # Paths
    # videos
    ogvideo_folder = args.ogvideo_folder
    # Flight record data downloaded from Airdata
    flight_folder = args.flight_folder
    # output of A
    metadata_path = args.metadata_categorized
    # output of this file 1
    resized_video_folder = args.resized_video_folder
    # output of this file 2
    detection_output_folder = args.detection_output_folder
    # output of this file 3
    summary_csv_path = args.summary_csv_path
    # output of Google Colab Code to train YOLO model
    model_path = args.model_path
    # output of this file 4
    frame_screencapture_folder = args.output_frame_screencapture_folder

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

if __name__ == "__main__":
    main()