import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from shapely.geometry import Polygon
from itertools import groupby, combinations
from collections import defaultdict

def get_video_filenames(directory, extensions=("mp4", "avi", "mov", "mkv")):
    video_files = [f for f in os.listdir(directory) if f.lower().endswith(extensions)]
    
    # Sort the files numerically if they contain numbers
    def extract_number(filename):
        match = re.search(r'\d+', filename)  # Find the first number in the filename
        return int(match.group()) if match else float('inf')  # Sort without numbers last

    video_files.sort(key=extract_number)
    return video_files

def read_flight_csv(filepath):
    data = pd.read_csv(filepath, header=0)
    return data

def filter_videos(df):
    return df[df["isVideo"] == 1]

# Easier data format
def extract_datetime_components(df, datetime_col="datetime_utc"):
    df = df.copy()  # Avoid modifying the original DataFrame
    df[datetime_col] = pd.to_datetime(df[datetime_col])  # Convert to datetime format

    # Extract components
    df["year"] = df[datetime_col].dt.year
    df["month"] = df[datetime_col].dt.month
    df["day"] = df[datetime_col].dt.day
    df["hour"] = df[datetime_col].dt.hour
    df["minute"] = df[datetime_col].dt.minute
    df["second"] = df[datetime_col].dt.second

    return df

# this is done for Airdata
def adjust_time(df):
    df = df.copy()  # Avoid modifying the original DataFrame
    df["hour"] += 8  # Add 8 hours
    df["day"] += (df["hour"] >= 24).astype(int)  # Increment day if hour >= 24
    df["hour"] %= 24  # Ensure hour remains within 0-23 range
    return df


# only needed columns
def select_columns(matching_photorows):
   
    selected_columns = [
        'time(millisecond)', 'datetime(utc)', 'latitude', 'longitude', 
        'height_above_takeoff(feet)', 'height_sonar(feet)', 'speed(mph)', 
        'distance(feet)', ' xSpeed(mph)', ' ySpeed(mph)', ' zSpeed(mph)', 
        ' compass_heading(degrees)', ' pitch(degrees)', ' roll(degrees)', 
        'gimbal_heading(degrees)', 'gimbal_pitch(degrees)', 'gimbal_roll(degrees)', 
        'altitude(feet)', 'ascent(feet)', 'videoname', 'year', 'month', 'day', 'hour', 'minute',
        'second'
    ]
    
    min_photorows = matching_photorows[selected_columns]
    
    return min_photorows

def add_pitch_and_rotation(df):
    df = df.copy()  # Avoid modifying the original DataFrame
    df["pitch"] = df["gimbal_pitch(degrees)"] / 180 * np.pi
    df["rotationfromnorth"] = df["gimbal_heading(degrees)"] / 180 * np.pi
    return df

def thetaData(all_corners):
    all_corners['theta'] = 90 - all_corners['gimbal_pitch(degrees)'].abs()
    return all_corners

def convert_ascentData(all_corners):
    all_corners['ascent(m)'] = all_corners['ascent(feet)'] * 0.3048
    return all_corners

def calculate_h(all_corners):
    all_corners['h'] = all_corners['ascent(m)'] / np.cos(np.radians(all_corners['theta']))
    return all_corners

def calculate_o(all_corners):
    all_corners['o'] = np.sqrt(all_corners['h']**2 - all_corners['ascent(m)']**2)
    return all_corners

def calculate_thetaphi(all_corners):
    all_corners['thetaphi'] = all_corners['theta'] + 24.366
    return all_corners

def calculate_theta_minus_fov_vv(df):
    df['theta_minus_fov_vv'] = df['theta'].apply(lambda theta: 24.366 - theta if theta < 24.366 else theta - 24.366)
    return df

def calculate_bdfd(all_corners):
    # Apply different calculations based on theta values
    mask = all_corners['theta'] > 24.366

    all_corners.loc[mask, 'bdfd'] = all_corners.loc[mask, 'ascent(m)'] * np.tan(np.radians(all_corners.loc[mask, 'theta_minus_fov_vv']))
    all_corners.loc[~mask, 'bdfd'] = all_corners.loc[~mask, 'ascent(m)'] * np.tan(np.radians(all_corners.loc[~mask, 'theta_minus_fov_vv']))

    return all_corners

def calculate_thetaphi_comp(all_corners):
    all_corners['thetaphi_comp'] = 90 - all_corners['thetaphi']
    all_corners['thetaphi_comp'] = all_corners['thetaphi_comp'].apply(lambda x: max(x, 5))
    return all_corners

def calculate_bd(all_corners):
    mask = all_corners['theta'] > 24.366
    
    all_corners.loc[mask, 'bd'] = all_corners.loc[mask, 'ascent(m)'] / np.tan(np.radians(all_corners.loc[mask, 'thetaphi_comp']))
    all_corners.loc[~mask, 'bd'] = all_corners.loc[~mask, 'ascent(m)'] * np.tan(np.radians(all_corners.loc[~mask, 'thetaphi']))

    return all_corners

def calculate_bdfdh(all_corners):
    mask = all_corners['theta'] > 24.366
    
    all_corners.loc[mask, 'bdfdh'] = all_corners.loc[mask, 'ascent(m)'] / np.cos(np.radians(all_corners.loc[mask, 'theta_minus_fov_vv']))
    all_corners.loc[~mask, 'bdfdh'] = all_corners.loc[~mask, 'bdfd'] / np.sin(np.radians(all_corners.loc[~mask, 'theta_minus_fov_vv']))

    return all_corners

def calculate_alt(all_corners):
    all_corners['alt'] = all_corners['bdfdh'] * np.cos(np.radians(24.366))
    return all_corners

def calculate_width(all_corners, sensor_width=8.89, focal_length=8.8):
    all_corners['width'] = sensor_width * all_corners['alt'] / focal_length
    return all_corners

def calculate_a1a2(all_corners):
    all_corners['a1a2'] = all_corners['width'] / 2
    return all_corners

def calculate_hypo(all_corners):
    all_corners['hypo'] = np.sqrt(all_corners['ascent(m)']**2 + all_corners['bd']**2)
    return all_corners

def calculate_seh(all_corners):
    all_corners['seh'] = np.sqrt(all_corners['bdfdh']**2 + all_corners['a1a2']**2)
    return all_corners

def calculate_db1(all_corners):
    all_corners['db1'] = (all_corners['hypo'] / all_corners['bdfdh']) * all_corners['seh']
    return all_corners

def calculate_b1b2(all_corners):
    all_corners['b1b2'] = np.sqrt(all_corners['db1']**2 - all_corners['hypo']**2)
    return all_corners

def add_frame_numbers(df):
    df = df.copy()  # Avoid modifying the original DataFrame
    df["frame_number"] = df.groupby("videoname").cumcount() + 1  # Assign frame numbers
    return df

# used in below function
def generate_video_corners(videoname, frame_number, rotation, a1a2, bdfd, o, b1b2, bd, tilt, theta):
    # limit for which drone position is no longer in frame extent on ground
    # derived from angle
    if theta > 24.366:
        y_values = [bd, bd, bdfd, bdfd, o, bd]
    else:
        y_values = [bd, bd, -bdfd, -bdfd, o, bd]

    cornersnotrec = pd.DataFrame({
        "corner": ["top_left", "top_right", "bottom_right", "bottom_left", "centre", "top_left"],
        "x": [-b1b2, b1b2, a1a2, -a1a2, 0, -b1b2],
        "y": y_values,
        "video": [videoname] * 6,
        "frameno": [frame_number] * 6,
        "rotation": [rotation] * 6,
        "tilt": [tilt] * 6
    })
    
    return cornersnotrec

# Create corners for all rows 
def create_all_video_corners(df):
    allcornersnotrec = pd.DataFrame(columns=["corner", "x", "y", "video", "frameno", "rotation", "tilt"])
    
    for i, row in df.iterrows():
        videoname = row["videoname"]
        frame_number = row["frame_number"]
        rotation = row["gimbal_heading(degrees)"]
        a1a2 = row["a1a2"]
        bdfd = row.get("bdfd", None)  # Handle missing columns safely
        o = row["o"]
        b1b2 = row["b1b2"]
        bd = row["bd"]
        tilt = row["gimbal_pitch(degrees)"]
        theta = row["theta"]

        # Generate corners based on theta condition
        cornersnotrec = generate_video_corners(videoname, frame_number, rotation, a1a2, bdfd, o, b1b2, bd, tilt, theta)

        # Append to main dataframe
        allcornersnotrec = pd.concat([allcornersnotrec, cornersnotrec], ignore_index=True)
    
    return allcornersnotrec

# sanity check function
def plot_proper_video_corners(allcornersnotrec):
    plt.figure(figsize=(8, 8))
    
    unique_videos = allcornersnotrec["video"].unique()
    
    for video in unique_videos:
        video_subset = allcornersnotrec[allcornersnotrec["video"] == video]
        unique_frames = video_subset["frameno"].unique()
        
        for frame in unique_frames:
            frame_subset = video_subset[video_subset["frameno"] == frame]
            
            # Extract the coordinates for the four corners (top_left, top_right, bottom_right, bottom_left)
            corners = frame_subset[["x", "y"]].iloc[:4].values
            # Close the polygon by adding the top_left at the end
            corners = np.vstack([corners, corners[0]])  
            plt.plot(corners[:, 0], corners[:, 1], marker='o', linestyle='-', color='black', alpha=0.7)
            
            # Plot the centre as a distinct point (red marker)
            centre = frame_subset.iloc[4]  # The 5th row is the centre
            plt.plot(centre["x"], centre["y"], marker='o', color='red', markersize=8)

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Proper Corners with Centre Marked")
    plt.gca().set_aspect('equal')  # Equivalent to coord_fixed(ratio = 1)
    plt.grid(True)
    
    plt.show()

# rotate corners based on orientation of drone
def rotate_all_video_corners(corners_df):
    
    corners_df = corners_df.copy()  

    corners_df["angleradians"] = np.radians(corners_df["rotation"])

    corners_df["xrot"] = corners_df["x"] * np.cos(corners_df["angleradians"]) + corners_df["y"] * np.sin(corners_df["angleradians"])
    corners_df["yrot"] = -corners_df["x"] * np.sin(corners_df["angleradians"]) + corners_df["y"] * np.cos(corners_df["angleradians"])

    return corners_df

# combine dataframe
def merge_video_corners_with_coordinates(corners_df, video_metadata_df):
   
    merged_df = corners_df.merge(
        video_metadata_df[["videoname", "frame_number", "latitude", "longitude"]],
        left_on=["Name", "frameno"],
        right_on=["videoname", "frame_number"],
        how="left"
    )
    
    return merged_df

# convert to lat lon
def calculate_video_rotated_latlon(df, meter_to_deg):
  
    df["latrot"] = df["latitude"] + (df["yrot"] * meter_to_deg)
    df["lonrot"] = df["longitude"] + (df["xrot"] * meter_to_deg) / np.cos(np.radians(df["latitude"]))
    
    return df

# sanity check function
def plot_video_rotated_corners_updated(df):

    plt.figure(figsize=(8, 6))
    
    for video, video_group in df.groupby("Name"):  
        for frame, frame_group in video_group.groupby("frameno"):
            # Extract the first 4 rows (corners)
            corners = frame_group[["lonrot", "latrot"]].iloc[:4].values
            corners = np.vstack([corners, corners[0]])  # Close the polygon
            
            plt.plot(corners[:, 0], corners[:, 1], marker='o', linestyle='-', color="blue", alpha=0.7)
            
            # Plot the centre as a distinct red point (5th row)
            centre = frame_group.iloc[4]  
            plt.plot(centre["lonrot"], centre["latrot"], marker='o', color='red', markersize=8)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Rotated Video Frame Corners with Centre")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


    import pandas as pd

# for frame fractions
def add_max_frame_number(viddetections, minvideorows):
    # Compute max frame number per 'Name'
    maxframes = minvideorows.groupby("videoname")["frame_number"].max().reset_index()
    maxframes.rename(columns={"frame_number": "max_frame_number"}, inplace=True)
    
    # Merge with viddetections (left join)
    merged_df = viddetections.merge(maxframes, on="videoname", how="left")
    
    return merged_df

# calculate frame fraction to match to closest frame
def normalize_and_round_frames(viddetections):
    # Ensure required columns exist
    if "frame_fraction" not in viddetections.columns or "max_frame_number" not in viddetections.columns:
        raise ValueError("Required columns 'frame_fraction' or 'max_frame_number' not found in dataframe.")
    
    # Compute frame_numeric safely
    viddetections["frame_numeric"] = viddetections["frame_fraction"] * viddetections["max_frame_number"]
    
    # Filter out invalid rows before rounding
    valid_mask = np.isfinite(viddetections["frame_numeric"])
    viddetections = viddetections[valid_mask].copy()
    
    # Round and convert to integer
    viddetections["frame_integer"] = np.round(viddetections["frame_numeric"]).astype(int)
    
    return viddetections


# Retaining top 40 percent of all detections by confidence
def filter_top_confidencev2(viddetections, threshold=0.6):

    print(f"Filtering detections with confidence >= {threshold}")
    viddetections = viddetections[viddetections["confidence"] >= threshold].reset_index(drop=True)
    return viddetections

# For plotting extent of drone path based on extent of all frames taken by drone
def get_video_hulls(rotatedvideocornerswithcoords):
    
    hull_data = {}

    unique_videos = rotatedvideocornerswithcoords["videoname"].unique()
    
    for video_name in unique_videos:
        # Filter data for the current video
        video_frames = rotatedvideocornerswithcoords[rotatedvideocornerswithcoords["videoname"] == video_name]
        
        # Remove rows with missing latitude or longitude
        video_frames = video_frames.dropna(subset=["latrot", "lonrot"])

        if video_frames.empty:
            continue
        
        # Extract lat/lon points
        points = video_frames[["lonrot", "latrot"]].values
        
        # Compute the convex hull
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]  # Get hull points
        
        # Ensure the hull closes by adding the first point at the end
        closed_hull_points = np.vstack([hull_points, hull_points[0]])

        # Store hull points for this video
        hull_data[video_name] = closed_hull_points

    return hull_data


def plot_all_hulls(hull_data):
  
    plt.figure(figsize=(10, 6))

    for video_name, hull_points in hull_data.items():
        plt.plot(hull_points[:, 0], hull_points[:, 1], marker='o', linestyle='-', label=video_name)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Convex Hulls of Video Frames")
    plt.legend()
    plt.grid(True)
    plt.show()

# Merging dataframes
def merge_video_detections_with_coordinates(detections, minvideorows):
    
    return detections.merge(
        minvideorows[["videoname", "frame_number", "gimbal_heading(degrees)", "latitude", "longitude", "ascent(m)"]],
        left_on=["videoname", "frame_integer"],  # Use frame_integer for the detections
        right_on=["videoname", "frame_number"],   
        how="left"
    )

# orient extent
def flip_y_coordinates(df):
    
    df[["y1", "y2", "y3", "y4"]] = 1 - df[["y1", "y2", "y3", "y4"]]
    return df

# Interpolate because of GSD for angled photos
def interpolate_y_boundsnew(y_norm, first_frame_data, video_name, frame_number):
    
    # Filter the first_frame_data DataFrame for the given video and frame number
    frame_subset = first_frame_data[first_frame_data["videoname"] == video_name]
    frame_subset = frame_subset[frame_subset["frameno"] == frame_number]

    # Extract y values based on the defined corners
    top_y = frame_subset.loc[frame_subset["corner"] == "top_left", "y"].values[0]
    bottom_y = frame_subset.loc[frame_subset["corner"] == "bottom_left", "y"].values[0]
    center_y = frame_subset.loc[frame_subset["corner"] == "centre", "y"].values[0]

    # Define corresponding normalized y values
    norm_y = np.array([0, 0.5, 1])  # Normalized range
    real_y = np.array([bottom_y, center_y, top_y])  # Real-world y values

    # Create interpolation function
    y_interpolator = interp1d(norm_y, real_y, fill_value="extrapolate")

    # Return interpolated real-world y-coordinate
    return y_interpolator(y_norm)

# Interpolate because of GSD for angled photos
def interpolate_x_boundsnew(y_val, trapezium_data):

    # Remove duplicate rows
    trapezium_data = trapezium_data.drop_duplicates()

    # Extract x and y values
    approx_x = trapezium_data["x"].values
    approx_y = trapezium_data["y"].values

    # Check if y_val is out of range
    if y_val < np.min(approx_y) or y_val > np.max(approx_y):
        print("Warning: y_val is outside the range of trapezium_data['y']; extrapolating.")

    # Interpolation for left side (x < 0)
    left_indices = approx_x < 0
    left_interpolator = interp1d(approx_y[left_indices], approx_x[left_indices], fill_value="extrapolate")
    x_min = left_interpolator(y_val)

    # Interpolation for right side (x > 0)
    right_indices = approx_x > 0
    right_interpolator = interp1d(approx_y[right_indices], approx_x[right_indices], fill_value="extrapolate")
    x_max = right_interpolator(y_val)

    return x_min, x_max

# used in bottom function
# to scale Bounding boxes after interpolation for frame extents has been applied
def scale_first_boxnew(detections, first_frame_data, interpolate_x_bounds, interpolate_y_bounds):
   
    def scale_row(row):
        video_name = row["videoname"]
        frame_number = row["frame_number"]

        # Filter the first_frame_data DataFrame for the given video and frame number
        filtered_frame = first_frame_data[
            (first_frame_data["videoname"] == video_name) & (first_frame_data["frame_number"] == frame_number)
        ]

        # Interpolate y-coordinates
        real_y1 = interpolate_y_bounds(row["y1"], filtered_frame, video_name, frame_number)
        real_y2 = interpolate_y_bounds(row["y2"], filtered_frame, video_name, frame_number)
        real_y3 = interpolate_y_bounds(row["y3"], filtered_frame, video_name, frame_number)
        real_y4 = interpolate_y_bounds(row["y4"], filtered_frame, video_name, frame_number)

        # Interpolate x-bounds using the real y-coordinates
        bounds1 = interpolate_x_bounds(real_y1, filtered_frame)
        bounds2 = interpolate_x_bounds(real_y2, filtered_frame)
        bounds3 = interpolate_x_bounds(real_y3, filtered_frame)
        bounds4 = interpolate_x_bounds(real_y4, filtered_frame)

        # Scale x-coordinates
        row["x1"] = row["x1"] * np.diff(bounds1)[0] + bounds1[0]
        row["x2"] = row["x2"] * np.diff(bounds2)[0] + bounds2[0]
        row["x3"] = row["x3"] * np.diff(bounds3)[0] + bounds3[0]
        row["x4"] = row["x4"] * np.diff(bounds4)[0] + bounds4[0]

        # Assign transformed y-coordinates
        row["y1"] = real_y1
        row["y2"] = real_y2
        row["y3"] = real_y3
        row["y4"] = real_y4

        return row

    # Apply function to each row
    scaled_detections = detections.apply(scale_row, axis=1)

    return scaled_detections

# using the previous three functions
def process_detections(detections, all_frame_data, interpolate_x_bounds, interpolate_y_bounds):
  
    all_scaled_detections = []

    # Iterate through each unique video name in the all_frame_data dataframe
    for video_name in all_frame_data["videoname"].unique():
        # Filter the detections DataFrame for the current video
        video_detections = detections[detections["videoname"] == video_name]

        # Check if there are any detections for this video
        if video_detections.empty:
            print(f"No detections for video: {video_name}")
            continue

        # Filter the all_frame_data DataFrame for the current video
        frame_data = all_frame_data[all_frame_data["videoname"] == video_name]

        # Apply the scaling function to the subset of detections for the current video
        scaled_detections = scale_first_boxnew(video_detections, frame_data, interpolate_x_bounds, interpolate_y_bounds)

        # Append the scaled detections to the list
        all_scaled_detections.append(scaled_detections)

    # Concatenate all the scaled detections into one DataFrame
    scaled_detections_df = pd.concat(all_scaled_detections, ignore_index=True)

    return scaled_detections_df

# matching to lat lon 
def create_boxes_scaledfinal_video(df, video_name=None):
    # Ensure the "VideoName" column exists before filtering
    if video_name is not None and "VideoName" in df.columns:
        df = df[df["VideoName"] == video_name]

    # Define scaling function
    def scale_row(row):
        return pd.Series({
            "x": [row["x1"], row["x2"], row["x3"], row["x4"], row["x1"]],
            "y": [row["y1"], row["y2"], row["y3"], row["y4"], row["y1"]],
            "longitude": row["longitude"],
            "latitude": row["latitude"]
        })

    # Apply function correctly
    scaled_df = df.apply(scale_row, axis=1)

    # Ensure "VideoName" column is added if present in the original dataframe
    if "videoname" in df.columns:
        scaled_df.insert(0, "videoname", df["videoname"].values)

    if "frame_number" in df.columns:
        scaled_df.insert(0, "frame_number", df["frame_number"].values)

    return scaled_df

# sanity check function
def plot_oriented_bboxes_for_video(scaled_df, videoname):
    # Filter for the selected video
    df_filtered = scaled_df[scaled_df["videoname"] == videoname]

    # Initialize plot
    fig, ax = plt.subplots(figsize=(8, 8))

    for _, row in df_filtered.iterrows():
        x_coords = list(row["x"])
        y_coords = list(row["y"])

        # Close the bounding box polygon by repeating the first point
        ax.plot(x_coords + [x_coords[0]], y_coords + [y_coords[0]], 'r-', linewidth=2)

    # Formatting
    ax.set_title(f"Oriented Bounding Boxes for Video: {videoname}")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_aspect('equal')  # Keep aspect ratio square
    plt.grid(True)
    plt.show()

# sanity check function
def plot_multiple_oriented_bboxes(scaled_df):
    # Iterate over each unique video name
    for videoname in scaled_df["videoname"].unique():
        plot_oriented_bboxes_for_video(scaled_df, videoname)

# match frame to correct videoname
def extract_videoname_and_frame(boxesscaledfinal):
    extracted_rows = []
    seen_frames = set()

    for _, row in boxesscaledfinal.iterrows():
        key = (row["videoname"], row["frame_number"])
        
        if key not in seen_frames:
            extracted_rows.append(key)
            seen_frames.add(key)

    return extracted_rows 

# key is Correct video and frame as both is unique but either is not
def get_frame_corners_for_unique_keys(extracted_rows, allvideocornersrotatedlatlon):
    # Initialize list to store the corner data for all frames
    all_corners = []

    # Loop through each unique (videoname, frame_number) pair in extracted_rows
    for videoname, frame_number in extracted_rows:
        # Filter the data for the given video and frame_number from allvideocornersrotatedlatlon
        frame_data = allvideocornersrotatedlatlon[(allvideocornersrotatedlatlon["videoname"] == videoname) & 
                                                  (allvideocornersrotatedlatlon["frame_number"] == frame_number)]

        # Check if there is enough data for this frame (at least 4 corners)
        if len(frame_data) < 4:
            raise ValueError(f"Not enough data for frame {frame_number} in video {videoname}")
        
        # Initialize a dictionary to store the corners for this frame
        frames_corners = {}
        
        # Extract lat/lon coordinates for the corners (top_left, top_right, bottom_right, bottom_left)
        corners = frame_data.iloc[:4][["latrot", "lonrot"]].values  # Get the first 4 rows

        # Store the corners for this frame
        frames_corners[frame_number] = [(corners[i][0], corners[i][1]) for i in range(4)]

        # Store the result in all_corners list
        all_corners.append((videoname, frame_number, frames_corners))

    return all_corners

# for below function
def extract_frames_by_video(data):
  
    video_frames = defaultdict(list)
    
    for video_name, frame_number, frame_data in data:
        video_frames[video_name].append((frame_number, frame_data[frame_number]))

    return video_frames

# calculate overlap between frame extents based on area
def calculate_overlap_percentage(frame1_corners, frame2_corners):
    
    if not frame1_corners or not frame2_corners:
        return 0  # Return 0 if any frame has no corners

    polygon1 = Polygon(frame1_corners)
    polygon2 = Polygon(frame2_corners)

    intersection_area = polygon1.intersection(polygon2).area
    smaller_area = min(polygon1.area, polygon2.area)

    return (intersection_area / smaller_area) * 100 if smaller_area != 0 else 0

# find clusters of frames - restarting when overlap is 10 percent - percentage can get low for angled frames
def find_clusters(video, frames_list):

    clusters = []
    i = 0  # Start with the first frame

    while i < len(frames_list) - 1:
        frame1_number, frame1_corners = frames_list[i]

        for j in range(i + 1, len(frames_list)):  # Iterate over the remaining frames
            frame2_number, frame2_corners = frames_list[j]
            overlap = calculate_overlap_percentage(frame1_corners, frame2_corners)

            if overlap < 90:  # If we found a break, start a new cluster
                clusters.append((video, frame1_number, frame2_number, "Cluster"))
                i = j  # Restart from this frame
                break
        else:
            # If no low-overlap frame is found, end with the last frame
            clusters.append((video, frame1_number, frames_list[-1][0], "Cluster"))
            break  # Stop iterating

    return clusters

# Using above functions
def process_video_frames(data):
    video_frames = extract_frames_by_video(data)
    selected_clusters = []

    for video, frames_list in video_frames.items():
        selected_clusters.extend(find_clusters(video, frames_list))

    return selected_clusters

# Clusters used to try to find concentrated areas of detections
def assign_clusters(boxesscaledfinal, clusters):
    
    # Ensure the Cluster column exists
    boxesscaledfinal["Cluster"] = None

    # Track cluster numbers per video
    cluster_counter = {}  # Dictionary to reset clusters for each video

    for video_name, start_frame, end_frame, _ in clusters:
        if video_name not in cluster_counter:
            cluster_counter[video_name] = 1  # Reset cluster numbering for each video
        
        # Assign cluster number to frames in the range [start_frame, end_frame - 1]
        mask = (
            (boxesscaledfinal["videoname"] == video_name) &
            (boxesscaledfinal["frame_number"] >= start_frame) &
            (boxesscaledfinal["frame_number"] < end_frame)  # Exclude last frame
        )
        boxesscaledfinal.loc[mask, "Cluster"] = cluster_counter[video_name]

        # Increment cluster number for this video
        cluster_counter[video_name] += 1

    boxesscaledfinal["Cluster"] = pd.to_numeric(boxesscaledfinal["Cluster"], errors="coerce")

    # Fill missing clusters using forward fill, grouped by video_name
    boxesscaledfinal["Cluster"] = boxesscaledfinal.groupby("videoname")["Cluster"].ffill()

    return boxesscaledfinal


# get all frames with at least one bounding box
# best filter so far - have tried frames with most bounding boxes or 3 BBs, but frame filtering is mostly arbitrary
def extract_frames_with_min_bboxes(boxesscaledfinal, min_bboxes=1):
    selected_rows = []

    grouped = boxesscaledfinal.groupby(["videoname", "Cluster"])

    for (video, cluster), group_df in grouped:
        # Count bounding boxes per frame
        frame_counts = group_df["frame_number"].value_counts()

        # Select frames with enough bounding boxes
        good_frames = frame_counts[frame_counts >= min_bboxes].index

        selected_rows.append(group_df[group_df["frame_number"].isin(good_frames)])

    return pd.concat(selected_rows, ignore_index=True)


def rotate_video_bounding_boxes(scaled_df, firstphoto):
    
    if firstphoto.empty:
        raise ValueError("firstphoto DataFrame is empty. Ensure the correct row is selected.")

    angle_rad = firstphoto['angleradians'].values[0]  # Extract rotation angle

    # Define a helper function to rotate each row
    def rotate_row(row):
        x_coords = np.array(row["x"])
        y_coords = np.array(row["y"])

        x_rot = x_coords * np.cos(angle_rad) + y_coords * np.sin(angle_rad)
        y_rot = -x_coords * np.sin(angle_rad) + y_coords * np.cos(angle_rad)

        return pd.Series({"xrot": x_rot.tolist(), "yrot": y_rot.tolist(),"longitude": row["longitude"], "latitude": row["latitude"]})

    # Apply rotation transformation to each row
    rotated_df = scaled_df.apply(rotate_row, axis=1)

    # If the original DataFrame has a Name column, add it back to the result
    for col in ["videoname", "frame_number", "Cluster"]:
        if col in scaled_df.columns:
            rotated_df.insert(0, col, scaled_df[col].values)
    return rotated_df

# discard frames if overlap is more than 70 percent, remove larger frame as smaller frames have better image resolution
def filter_frames_by_overlap(frames_cornersr2, extracted_rowsr2):
    to_remove = set()

    # Iterate over all unique combinations of frames
    for (file1, frame1, corners_dict1), (file2, frame2, corners_dict2) in combinations(frames_cornersr2, 2):
        if (file1, frame1) in to_remove or (file2, frame2) in to_remove:
            continue  # Skip if either frame is already marked for removal
        
        # Extract corner coordinates
        corners1 = list(corners_dict1.values())[0]  # Get the corners for frame1
        corners2 = list(corners_dict2.values())[0]  # Get the corners for frame2
        
        # Calculate the overlap percentage
        overlap = calculate_overlap_percentage(corners1, corners2)

        if overlap > 70:
            # Remove the frame with larger area
            area1 = Polygon(corners1).area
            area2 = Polygon(corners2).area

            if area1 > area2:
                to_remove.add((file1, frame1))
            else:
                to_remove.add((file2, frame2))

    # Filter the frames list by removing frames in 'to_remove'
    filtered_frames = [frame for frame in frames_cornersr2 if (frame[0], frame[1]) not in to_remove]

    # If extracted_rowsr2 is a list, filter it similarly by removing frames in 'to_remove'
    filtered_extracted_rows = [row for i, row in enumerate(extracted_rowsr2) if (frames_cornersr2[i][0], frames_cornersr2[i][1]) not in to_remove]

    return filtered_frames, filtered_extracted_rows

# rotate BBs - frames were previously rotated
def rotate_bboxes_for_videos(scaled_df, all_corners_df):

    rotated_bboxes_list = []

    for (videoname, frame_number) in scaled_df[["videoname", "frame_number"]].drop_duplicates().itertuples(index=False):
        firstphoto = all_corners_df[(all_corners_df["videoname"] == videoname) & 
                                    (all_corners_df["frame_number"] == frame_number)]
        
        if firstphoto.empty:
            print(f"Warning: No data found in all_corners_df for {videoname}, frame {frame_number}. Skipping rotation.")
            continue
        
        # Filter scaled_df for the specific video and frame
        scaled_df_filtered = scaled_df[(scaled_df["videoname"] == videoname) & 
                                       (scaled_df["frame_number"] == frame_number)]

        # Rotate bounding boxes
        rotated_bboxes = rotate_video_bounding_boxes(scaled_df_filtered, firstphoto)
        
        # Store results
        rotated_bboxes_list.append(rotated_bboxes)

    # Combine results
    final_rotated_df = pd.concat(rotated_bboxes_list, ignore_index=True)

    return final_rotated_df

# sanity check function
def plot_all_rotated_bounding_boxes_video(rotated_df):

    fig, ax = plt.subplots(figsize=(8, 8))  # Create a single figure

    # Assign different colors for each video-frame combination
    unique_keys = rotated_df.groupby(["videoname", "frame_number"]).ngroups
    cmap = plt.get_cmap("tab10")  # Choose a colormap
    color_map = {}  # Store assigned colors

    # Plot each rotated bounding box
    for (videoname, frame_number), df_filtered in rotated_df.groupby(["videoname", "frame_number"]):
        if (videoname, frame_number) not in color_map:
            color_map[(videoname, frame_number)] = cmap(len(color_map) % 10)  # Assign unique color

        for _, row in df_filtered.iterrows():
            x_coords = row["xrot"] + [row["xrot"][0]]  # Close the bounding box
            y_coords = row["yrot"] + [row["yrot"][0]]
            ax.plot(x_coords, y_coords, marker='o', linestyle='-', color=color_map[(videoname, frame_number)], 
                    label=f"{videoname} - Frame {frame_number}" if (videoname, frame_number) not in color_map else "")

    # Formatting
    ax.set_xlabel("X (Rotated)")
    ax.set_ylabel("Y (Rotated)")
    ax.set_title("Rotated Bounding Boxes for All Videos and Frames")
    ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# convert BBs to lat lon
def convert_to_real_world_coordinates(rotated_df):
   
    rotated_df = rotated_df.copy()

    # Define conversion factor (approximate, varies slightly with latitude)
    metertodeg = 1 / 111320  # Degrees per meter

    # Function to apply element-wise transformations
    def transform_row(row):
        latrot = row["latitude"] + (np.array(row["yrot"]) * metertodeg)
        lonrot = row["longitude"] + (np.array(row["xrot"]) * metertodeg) / np.cos(np.radians(row["latitude"]))
        return pd.Series({"latrot": latrot, "lonrot": lonrot})

    # Apply the transformation row-wise
    rotated_df[["latrot", "lonrot"]] = rotated_df.apply(transform_row, axis=1)

    return rotated_df

# sanity check function
def plot_video_frame_extents(rotatedcornerswithcoords, realworldframes, title="Rotated Video Frame Extent and Bounding Boxes"):

    fig, ax = plt.subplots(figsize=(24, 4))  # Create a single figure and axis

    # Get unique video frames that exist in realworldframes
    unique_video_frames = realworldframes[["videoname", "frame_number"]].drop_duplicates()

    for _, (videoname, frame_number) in unique_video_frames.iterrows():
        # Get the extent (corners) for the current video frame
        corners_frame = rotatedcornerswithcoords[
            (rotatedcornerswithcoords["videoname"] == videoname) & 
            (rotatedcornerswithcoords["frame_number"] == frame_number)
        ]

        # Get the detections (bounding boxes) for the current video frame
        rotated_detections = realworldframes[
            (realworldframes["videoname"] == videoname) & 
            (realworldframes["frame_number"] == frame_number)
        ]

        # Ensure there are at least 5 rows (4 corners + center)
        if len(corners_frame) < 5:
            continue  # Skip if data is insufficient

        # Plot video frame extent (excluding the center point)
        lat_extent = corners_frame["latrot"].iloc[:4].tolist() + [corners_frame["latrot"].iloc[0]]  
        lon_extent = corners_frame["lonrot"].iloc[:4].tolist() + [corners_frame["lonrot"].iloc[0]]
        ax.plot(lon_extent, lat_extent, marker='o', linestyle='--', color="black", 
                label=f"{videoname} - Frame {frame_number} Extent")

        # Plot bounding boxes in lat/lon
        for _, row in rotated_detections.iterrows():
            if isinstance(row["latrot"], list) and isinstance(row["lonrot"], list):
                lat_coords = row["latrot"] + [row["latrot"][0]]  # Close the bounding box
                lon_coords = row["lonrot"] + [row["lonrot"][0]]
            else:
                lat_coords = np.array(row["latrot"]).tolist() + [row["latrot"][0]]  
                lon_coords = np.array(row["lonrot"]).tolist() + [row["lonrot"][0]]

            ax.plot(lon_coords, lat_coords, marker='o', linestyle='-', label=f"{videoname} - Frame {frame_number}" if _ == 0 else "")

        # Plot center as a red point
        centre = corners_frame.iloc[4]  # Assuming the 5th row is the center point
        ax.plot(centre["lonrot"], centre["latrot"], marker='o', color='red', markersize=8)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.show()

# based on key, which is video and frame
def filter_selected_detections_by_frames(filtered_extracted_rows, selecteddetections):
    frame_keys = set((videoname, frame_number) for videoname, frame_number in filtered_extracted_rows)
    
    # Filter selecteddetections to keep only the rows where (videoname, frame_number) are in frame_keys
    filtered_selecteddetections = selecteddetections[selecteddetections.apply(
        lambda row: (row['videoname'], row['frame_number']) in frame_keys, axis=1
    )]
    
    return filtered_selecteddetections

# filter bounding boxes by overlap too, keep smaller bounding boxes
def filter_by_overlapdetectionsfinal(dataframe, overlap_threshold=70):
    to_remove = set()  # Use a set to avoid duplicate indices

    for index, row in dataframe.iterrows():
        if index in to_remove:
            continue  # Skip rows already marked for removal

        lat_rot = row['latrot']
        lon_rot = row['lonrot']
        frame_corners = list(zip(lon_rot, lat_rot))
        polygon1 = Polygon(frame_corners)

        for compare_index, compare_row in dataframe.iterrows():
            if index == compare_index or compare_index in to_remove:
                continue

            compare_lat_rot = compare_row['latrot']
            compare_lon_rot = compare_row['lonrot']
            compare_frame_corners = list(zip(compare_lon_rot, compare_lat_rot))
            polygon2 = Polygon(compare_frame_corners)

            overlap = calculate_overlap_percentage(frame_corners, compare_frame_corners)

            if overlap > overlap_threshold:
                # 🔄 Reverse logic to remove the bigger polygon instead
                if polygon1.area > polygon2.area:
                    to_remove.add(index)
                else:
                    to_remove.add(compare_index)
                break  # Break after marking one for removal

    # Drop rows by index
    filtered_dataframe = dataframe.drop(index=to_remove)

    return filtered_dataframe

# metadata used for analysis 
def export_final_detection_summary(final_detections_df, output_path):
    
    # Count bounding boxes per video and frame
    box_counts = (
        final_detections_df.groupby(["videoname", "frame_number"])
        .size()
        .reset_index(name="bbox_count")
    )
    
    # Merge confidence values - preserving one row per bbox
    confidence_df = final_detections_df[["videoname", "frame_number"]]
    
    # Merge counts back to confidence dataframe
    merged_df = confidence_df.merge(box_counts, on=["videoname", "frame_number"], how="left")
    
    merged_df = merged_df.sort_values(by=["videoname", "frame_number"]).reset_index(drop=True)
    
    # Export to CSV
    merged_df.to_csv(output_path, index=False)
    print(f"Summary CSV exported to: {output_path}")