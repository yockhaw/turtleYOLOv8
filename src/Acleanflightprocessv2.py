import os
import pandas as pd
from functions.processingfuncs import *
import argparse

# This is the first file to run
# Put the data downloaded from Airdata into the flight_folder path

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Input video folder, flight data folder and output folder"
    )

    parser.add_argument(
        "--video_folder",
        type=str,
        required=True
    )

    parser.add_argument(
        "--flight_folder",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output_metadata",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output_metadata_categorized",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output_frames",
        type=str,
        required=True
    )

    return parser.parse_args()

def main():
    args = parse_arguments()
    # videos
    video_folder = args.video_folder
    # flight records from Airdata
    flight_folder = args.flight_folder
    output_folder = args.output_folder
    
    os.makedirs(output_folder, exist_ok=True)

    print("Adding timezone-corrected local time to flight record files...")
    add_local_time_to_flight_records(flight_folder)  
    
    print("Extracting video metadata...")
    df_metadata = extract_video_metadata(video_folder)
    df_metadata.to_csv(args.output_metadata, index=False)
    
    print("Cleaning and filtering metadata...")
    df_filtered = clean_metadata(df_metadata)
    print(df_filtered)
    
    # using timezone to calculate local time, useful later
    print("Computing time and timezone info...")
    time_info = df_filtered.apply(compute_time_info, axis=1)
    df_filtered = pd.concat([df_filtered, time_info], axis=1)
    
    print("Loading flight data...")
    flight_times = load_flight_data(flight_folder)
    
    # As every flight can have multiple videos taken, it is important to match each video to whichever flight record it belongs it based on time taken
    print("Matching videos to flight records...")
    df_filtered = match_videos_to_flights(df_filtered, flight_times)
    df_filtered.to_csv(args.output_metadata_categorized, index=False)
    
    # As resolution of flight record is 100 ms, a 30 second flight will yield 300 rows of data - this is to keep track of frames in each video
    print("Extracting flight data per video...")
    extract_flight_data_per_video(df_filtered, flight_folder, output_folder, args.output_frames)
    
    print("Processing complete.")

if __name__ == "__main__":
    main()