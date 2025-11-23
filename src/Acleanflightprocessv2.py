import os
import pandas as pd
from functions.processingfuncs import *

# This is the first file to run
# Put the data downloaded from Airdata into the flight_folder path

def main():
    # videos
    video_folder = '/Volumes/External SS/cleanvideos'
    # flight records from Airdata
    flight_folder = '/Users/fooyockhaw/Desktop/allflightdata'
    output_folder = '/Users/fooyockhaw/Desktop/flightforuniquevideosv3'
    
    os.makedirs(output_folder, exist_ok=True)

    print("Adding timezone-corrected local time to flight record files...")
    add_local_time_to_flight_records(flight_folder)  
    
    print("Extracting video metadata...")
    df_metadata = extract_video_metadata(video_folder)
    df_metadata.to_csv('/Users/fooyockhaw/Desktop/video_metadatav3.csv', index=False)
    
    print("Cleaning and filtering metadata...")
    df_filtered = clean_metadata(df_metadata)
    
    # using timezone to calculate local time, useful later
    print("Computing time and timezone info...")
    time_info = df_filtered.apply(compute_time_info, axis=1)
    df_filtered = pd.concat([df_filtered, time_info], axis=1)
    
    print("Loading flight data...")
    flight_times = load_flight_data(flight_folder)
    
    # As every flight can have multiple videos taken, it is important to match each video to whichever flight record it belongs it based on time taken
    print("Matching videos to flight records...")
    df_filtered = match_videos_to_flights(df_filtered, flight_times)
    df_filtered.to_csv('/Users/fooyockhaw/Desktop/video_metadata_categorizedv3.csv', index=False)
    
    # As resolution of flight record is 100 ms, a 30 second flight will yield 300 rows of data - this is to keep track of frames in each video
    print("Extracting flight data per video...")
    extract_flight_data_per_video(df_filtered, flight_folder, output_folder, '/Users/fooyockhaw/Desktop/rows_extracted_summary.csv')
    
    print("Processing complete.")

if __name__ == "__main__":
    main()