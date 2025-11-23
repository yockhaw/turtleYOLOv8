import os
import pandas as pd
import pytz
from datetime import datetime, timedelta
from timezonefinder import TimezoneFinder
from exiftool import ExifToolHelper

def extract_video_metadata(video_folder):
    all_metadata = []
    with ExifToolHelper() as et:
        for file_name in os.listdir(video_folder):
            if not file_name.lower().endswith(('.mov', '.mp4', '.avi')):
                continue
            file_path = os.path.join(video_folder, file_name)
            try:
                meta_list = et.get_metadata(file_path)
                for meta in meta_list:
                    meta['FileName'] = file_name
                    all_metadata.append(meta)
            except Exception as e:
                print(f"Failed to get metadata for {file_path}: {e}")
    return pd.DataFrame(all_metadata)

def clean_metadata(df_metadata):
    df_filtered = df_metadata[
        ['File:FileName', 'File:Directory', 'QuickTime:CreateDate', 'QuickTime:Duration', 'Composite:GPSLatitude', 'Composite:GPSLongitude']
    ].rename(columns={
        'File:FileName': 'FileName',
        'File:Directory': 'Directory',
        'QuickTime:CreateDate': 'CreateDate',
        'QuickTime:Duration': 'Duration',
        'Composite:GPSLatitude': 'Latitude',
        'Composite:GPSLongitude': 'Longitude'
    })
    return df_filtered

def get_timezone(lat, lon, tf=TimezoneFinder()):
    tz = tf.timezone_at(lat=lat, lng=lon)
    if tz is None:
        raise ValueError(f"Could not determine timezone for lat={lat}, lon={lon}")
    return tz

def compute_time_info(row):
    tz_str = get_timezone(row['Latitude'], row['Longitude'])
    tz = pytz.timezone(tz_str)
    create_naive = datetime.strptime(row['CreateDate'], '%Y:%m:%d %H:%M:%S')
    create_local = tz.localize(create_naive)
    end_local = create_local + timedelta(seconds=float(row['Duration']))
    return pd.Series({'CreateDate_local': create_local, 'EndDate': end_local, 'Timezone': tz_str})

def load_flight_data(flight_folder):
    flight_times = []
    for f in os.listdir(flight_folder):
        if f.endswith('.csv'):
            df_flight = pd.read_csv(os.path.join(flight_folder, f))
            df_flight['datetime(utc)'] = pd.to_datetime(df_flight['datetime(utc)'], utc=True)
            flight_times.append({'file': f, 'datetimes': df_flight['datetime(utc)']})
    return flight_times

def match_videos_to_flights(df_metadata_filtered, flight_times):
    flight_records, trim_flags = [], []
    for _, row in df_metadata_filtered.iterrows():
        matched_flight, needs_trim = None, False
        video_tz = pytz.timezone(row['Timezone'])
        for flight in flight_times:
            datetimes_local = flight['datetimes'].dt.tz_convert(video_tz)
            flight_start, flight_end = datetimes_local.iloc[0], datetimes_local.iloc[-1]
            if flight_start <= row['CreateDate_local'] <= flight_end:
                matched_flight = flight['file']
                if row['EndDate'] > flight_end:
                    needs_trim = True
                break
        flight_records.append(matched_flight)
        trim_flags.append(needs_trim)
    df_metadata_filtered['FlightRecord'] = flight_records
    df_metadata_filtered['NeedsTrim'] = trim_flags
    return df_metadata_filtered

def extract_flight_data_per_video(df_metadata_filtered, flight_folder, output_folder, summary_csv_path):
    processed_results = []
    summary_records = []

    for flight_file in df_metadata_filtered['FlightRecord'].dropna().unique():
        flight_path = os.path.join(flight_folder, flight_file)
        df_flight = pd.read_csv(flight_path)
        df_flight['datetime(utc)'] = pd.to_datetime(df_flight['datetime(utc)'], utc=True)
        
        videos_in_flight = df_metadata_filtered[
            df_metadata_filtered['FlightRecord'] == flight_file
        ]
        
        if videos_in_flight.empty:
            continue
        
        for _, video_row in videos_in_flight.iterrows():
            video_tz = pytz.timezone(video_row['Timezone'])
            df_flight['datetime_local'] = df_flight['datetime(utc)'].dt.tz_convert(video_tz)
            
            start_time = video_row['CreateDate_local']
            
            if not video_row['NeedsTrim']:
                end_time = video_row['EndDate']
            else:
                end_time = df_flight['datetime_local'].iloc[-1]
            
            flight_rows_in_window = df_flight[
                (df_flight['datetime_local'] >= start_time.floor('S')) & 
                (df_flight['datetime_local'] <= end_time.ceil('S'))
            ].copy()
            
            if flight_rows_in_window.empty:
                print(f"No matching rows for video {video_row['FileName']} in flight {flight_file}")
                rows_extracted = 0
                full_extract = pd.DataFrame()  # Empty DataFrame
            else:
                if not video_row['NeedsTrim']:
                    # Usual extraction with start, middle, end rows
                    rows_start_second = flight_rows_in_window[
                        (flight_rows_in_window['datetime_local'] >= start_time.floor('S')) & 
                        (flight_rows_in_window['datetime_local'] < start_time.floor('S') + timedelta(seconds=1))
                    ].tail(5)
                    
                    rows_end_second = flight_rows_in_window[
                        (flight_rows_in_window['datetime_local'] >= end_time.floor('S')) & 
                        (flight_rows_in_window['datetime_local'] < end_time.floor('S') + timedelta(seconds=1))
                    ].head(5)
                    
                    rows_middle = flight_rows_in_window[
                        (flight_rows_in_window['datetime_local'] >= start_time.floor('S') + timedelta(seconds=1)) & 
                        (flight_rows_in_window['datetime_local'] < end_time.floor('S'))
                    ]
                    
                    full_extract = pd.concat([rows_start_second, rows_middle, rows_end_second], ignore_index=True)
                
                else:
                    # For trimmed videos, extract everything from start_time to last row
                    full_extract = flight_rows_in_window.reset_index(drop=True)
                
                rows_extracted = len(full_extract)
                output_path = os.path.join(output_folder, f"{video_row['FileName'].rsplit('.', 1)[0]}_extract.csv")
                full_extract.to_csv(output_path, index=False)
                
                processed_results.append({
                    'FlightRecord': flight_file,
                    'VideoFile': video_row['FileName'],
                    'ExtractedRows': full_extract
                })
                
                print(f"Processed video {video_row['FileName']} - total rows extracted: {rows_extracted}")

            # Add to summary regardless
            summary_records.append({
                'VideoName': video_row['FileName'],
                'RowsExtracted': rows_extracted,
                'Duration': video_row['Duration'],
                'NeedsTrim': video_row['NeedsTrim']
            })
    
    # Save summary
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary saved to {summary_csv_path}")

    return processed_results


def add_local_time_to_flight_records(flight_folder, output_folder=None):
    """
    Adds a 'datetime_local' column to each flight record CSV based on estimated timezone from coordinates.
    """
    tf = TimezoneFinder()

    for file_name in os.listdir(flight_folder):
        if not file_name.lower().endswith('.csv'):
            continue

        file_path = os.path.join(flight_folder, file_name)
        print(f"Processing {file_name}...")

        df = pd.read_csv(file_path)
        
        if 'datetime(utc)' not in df.columns:
            print(f"Skipping {file_name}, no 'datetime(utc)' column found.")
            continue

        # Ensure datetime(utc) is in datetime format
        df['datetime(utc)'] = pd.to_datetime(df['datetime(utc)'], utc=True)

        # Get timezone using lat/lon
        if 'latitude' in df.columns and 'longitude' in df.columns:
            lat = df['latitude'].iloc[0]
            lon = df['longitude'].iloc[0]
        elif 'Latitude' in df.columns and 'Longitude' in df.columns:
            lat = df['Latitude'].iloc[0]
            lon = df['Longitude'].iloc[0]
        else:
            print(f"Skipping {file_name}, no latitude/longitude columns found.")
            continue

        timezone_str = tf.timezone_at(lat=lat, lng=lon)
        if timezone_str is None:
            print(f"Could not determine timezone for {file_name}, skipping.")
            continue

        tz = pytz.timezone(timezone_str)
        df['datetime_local'] = df['datetime(utc)'].dt.tz_convert(tz)

        # output
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, file_name)
        else:
            output_path = file_path  # overwrite if needed

        df.to_csv(output_path, index=False)
        print(f"Saved updated file to {output_path}")