import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

def read_flight_csv(filepath):
    data = pd.read_csv(filepath, header=0)
    return data

def extractIsPhoto(df):
    return df[df["isPhoto"] == 1]

def convert_to_umt8(df):
    df["datetime(utc)"] = pd.to_datetime(df["datetime(utc)"])
    df["datetime(umt8)"] = df["datetime(utc)"] + pd.Timedelta(hours=8)
    df["year"] = df["datetime(umt8)"].dt.year
    df["month"] = df["datetime(umt8)"].dt.month
    df["day"] = df["datetime(umt8)"].dt.day
    df["hour"] = df["datetime(umt8)"].dt.hour
    df["minute"] = df["datetime(umt8)"].dt.minute
    df["second"] = df["datetime(umt8)"].dt.second
    
    return df

def extractqgistime(df, timesplit_column):
    df[['hour', 'minute', 'second']] = df[timesplit_column].str.split(':', expand=True)
    
    df['hour'] = pd.to_numeric(df['hour'], errors='coerce')
    df['minute'] = pd.to_numeric(df['minute'], errors='coerce')
    df['second'] = pd.to_numeric(df['second'], errors='coerce')
    
    return df

def find_best_match(row, photorows):
    match = photorows[
        (photorows['hour'] == row['hour']) &
        (photorows['minute'] == row['minute']) &
        (photorows['second'] == row['second'])
    ]

    if match.empty:
        match = photorows[
            (photorows['hour'] == row['hour']) &
            (photorows['minute'] == row['minute']) &
            (photorows['second'] == row['second'] + 1)
        ]
    if not match.empty:
        return match.iloc[0]
    else:
        return None
    
def get_matched_rows(qgis, photorows):
    matched_rows = []  
    
    for _, qgis_row in qgis.iterrows():
        matched_row = find_best_match(qgis_row, photorows)  
        if matched_row is not None:
            matched_rows.append(matched_row) 
    
    matched_df = pd.DataFrame(matched_rows)
    
    return matched_df

def arrange_by_time(thphoto):

    sorted_thphoto = thphoto.sort_values(by=['hour', 'minute', 'second'])
    return sorted_thphoto

def append_columns_to_dataframe(target_df, source_df, columns):
    
    missing_columns = [col for col in columns if col not in source_df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in source DataFrame: {', '.join(missing_columns)}")

    # Append columns from source_df to target_df
    for col in columns:
        target_df[col] = source_df[col].values
    
    return target_df

def select_columns(matching_photorows):
   
    selected_columns = [
        'time(millisecond)', 'datetime(utc)', 'latitude', 'longitude', 
        'height_above_takeoff(feet)', 'height_sonar(feet)', 'speed(mph)', 
        'distance(feet)', ' xSpeed(mph)', ' ySpeed(mph)', ' zSpeed(mph)', 
        ' compass_heading(degrees)', ' pitch(degrees)', ' roll(degrees)', 
        'gimbal_heading(degrees)', 'gimbal_pitch(degrees)', 'gimbal_roll(degrees)', 
        'altitude(feet)', 'ascent(feet)', 'year', 'month', 'day', 'hour', 'minute',
        'second', 'Name', 'Altitude', 'Path'
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

def generate_corners(photo_name, rotation, a1a2, bdfd, o, b1b2, bd, tilt, theta):
    if theta > 24.366:
        y_values = [bd, bd, bdfd, bdfd, o, bd]
    else:
        y_values = [bd, bd, -bdfd, -bdfd, o, bd]

    cornersnotrec = pd.DataFrame({
        "corner": ["top_left", "top_right", "bottom_right", "bottom_left", "centre", "top_left"],
        "x": [-b1b2, b1b2, a1a2, -a1a2, 0, -b1b2],
        "y": y_values,
        "photo": [photo_name] * 6,
        "rotation": [rotation] * 6,
        "tilt": [tilt] * 6
    })
    
    return cornersnotrec

def create_all_corners(df):
    """Create corners for all rows in the dataframe."""
    allcornersnotrec = pd.DataFrame(columns=["corner", "x", "y", "photo", "rotation", "tilt"])
    
    for i, row in df.iterrows():
        photo_name = row["Name"]
        rotation = row["gimbal_heading(degrees)"]
        a1a2 = row["a1a2"]
        bdfd = row.get("bdfd", None)  # Handle missing columns safely
        o = row["o"]
        b1b2 = row["b1b2"]
        bd = row["bd"]
        tilt = row["gimbal_pitch(degrees)"]
        theta = row["theta"]

        # Generate corners based on theta condition
        cornersnotrec = generate_corners(photo_name, rotation, a1a2, bdfd, o, b1b2, bd, tilt, theta)

        # Append to main dataframe
        allcornersnotrec = pd.concat([allcornersnotrec, cornersnotrec], ignore_index=True)
    
    return allcornersnotrec

def plot_proper_corners(allcornersnotrec):
    
    plt.figure(figsize=(8, 8))
    
    unique_photos = allcornersnotrec["Name"].unique()
    
    for Name in unique_photos:
        subset = allcornersnotrec[allcornersnotrec["Name"] == Name]
        
        # Extract the coordinates for the four corners (top_left, top_right, bottom_right, bottom_left)
        corners = subset[["x", "y"]].iloc[:4].values
        # Close the polygon by adding the top_left at the end
        corners = np.vstack([corners, corners[0]])  
        plt.plot(corners[:, 0], corners[:, 1], marker='o', linestyle='-', color='black', alpha=0.7)
        
        # Plot the centre as a distinct point (red marker)
        centre = subset[subset["x"] == subset["x"].iloc[4]]  # Assuming 'centre' is the 5th row
        plt.plot(centre["x"], centre["y"], marker='o', color='red', markersize=8)
    
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Proper Corners with Centre Marked")
    plt.gca().set_aspect('equal')  # Equivalent to coord_fixed(ratio = 1)
    plt.grid(True)
    
    plt.show()

def rotate_all_corners(corners_df):
    
    corners_df = corners_df.copy()  

    corners_df["angleradians"] = np.radians(corners_df["rotation"])

    corners_df["xrot"] = corners_df["x"] * np.cos(corners_df["angleradians"]) + corners_df["y"] * np.sin(corners_df["angleradians"])
    corners_df["yrot"] = -corners_df["x"] * np.sin(corners_df["angleradians"]) + corners_df["y"] * np.cos(corners_df["angleradians"])

    return corners_df

def merge_corners_with_coordinates(all_corners_rotated, minphotorows):
    
    return all_corners_rotated.merge(
        minphotorows[["Name", "latitude", "longitude"]],
        on="Name",  
        how="left"
    )

def calculate_rotated_latlon(df, meter_to_deg):
  
    df["latrot"] = df["latitude"] + (df["yrot"] * meter_to_deg)
    df["lonrot"] = df["longitude"] + (df["xrot"] * meter_to_deg) / np.cos(np.radians(df["latitude"]))
    
    return df

def plot_rotated_cornersUpdated(df):

    plt.figure(figsize=(8, 6))
    
    for name, group in df.groupby("Name"):  
        # Extract the first 4 rows (corners) and plot them
        corners = group[["lonrot", "latrot"]].iloc[:4].values
        # Close the polygon by adding the first point again at the end
        corners = np.vstack([corners, corners[0]])  # Ensures the polygon is closed
        plt.plot(corners[:, 0], corners[:, 1], marker='o', linestyle='-', color="blue", label=name)
        
        # Plot the centre as a distinct red point (excluding it from the polygon)
        centre = group.iloc[4]  # Assuming the 5th row is the centre point
        plt.plot(centre["lonrot"], centre["latrot"], marker='o', color='red', markersize=8)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Rotated Photo Corners with Centre")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

def merge_detections_with_coordinates(detections, minphotorows):
    
    return detections.merge(
        minphotorows[["Name", "gimbal_heading(degrees)", "latitude", "longitude", "Altitude"]],
        on="Name",  
        how="left"
    )

def flip_y_coordinates(df):
    
    df[["y1", "y2", "y3", "y4"]] = 1 - df[["y1", "y2", "y3", "y4"]]
    return df

def interpolate_y_boundsnew(y_norm, first_photo, image_name):
   
    # Filter the first_photo DataFrame for the given image
    photo_subset = first_photo[first_photo["Name"] == image_name]

    # Extract y values based on the defined corners
    top_y = photo_subset.loc[photo_subset["corner"] == "top_left", "y"].values[0]  # or top_right
    bottom_y = photo_subset.loc[photo_subset["corner"] == "bottom_left", "y"].values[0]  # or bottom_right
    center_y = photo_subset.loc[photo_subset["corner"] == "centre", "y"].values[0]

    # Define corresponding normalized y values
    norm_y = np.array([0, 0.5, 1])  # Normalized range
    real_y = np.array([bottom_y, center_y, top_y])  # Real-world y values

    # Create interpolation function
    y_interpolator = interp1d(norm_y, real_y, fill_value="extrapolate")

    # Return interpolated real-world y-coordinate
    return y_interpolator(y_norm)


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

def scale_first_boxnew(first_box, first_photo, interpolate_x_bounds, interpolate_y_bounds):
    
    def scale_row(row):
        # Automatically get the image name from the row
        image_name = row["Name"]

        # Filter the first_photo DataFrame for the given image
        filtered_photo = first_photo[first_photo["Name"] == image_name]

        # Interpolate y-coordinates
        real_y1 = interpolate_y_bounds(row["y1"], filtered_photo, image_name)
        real_y2 = interpolate_y_bounds(row["y2"], filtered_photo, image_name)
        real_y3 = interpolate_y_bounds(row["y3"], filtered_photo, image_name)
        real_y4 = interpolate_y_bounds(row["y4"], filtered_photo, image_name)

        # Interpolate x-bounds using the real y-coordinates
        bounds1 = interpolate_x_bounds(real_y1, filtered_photo)
        bounds2 = interpolate_x_bounds(real_y2, filtered_photo)
        bounds3 = interpolate_x_bounds(real_y3, filtered_photo)
        bounds4 = interpolate_x_bounds(real_y4, filtered_photo)

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
    first_box_scaled = first_box.apply(scale_row, axis=1)
    
    return first_box_scaled

def process_detections(detectionswithdims, allcornersrotatedlatlon, interpolate_x_bounds, interpolate_y_bounds):
    # Initialize an empty list to store the scaled results
    all_scaled_detections = []

    # Iterate through each unique photo name in the allcornersrotatedlatlon dataframe
    for image_name in allcornersrotatedlatlon["Name"].unique():
        # Filter the detectionswithdims DataFrame for the current image name
        detections_subset = detectionswithdims[detectionswithdims["Name"] == image_name]

        # Check if there are any detections for this image
        if detections_subset.empty:
            # If no detections, skip this photo
            print(f"No detections for photo: {image_name}")
            continue

        # Filter the allcornersrotatedlatlon DataFrame for the current image name
        photo_subset = allcornersrotatedlatlon[allcornersrotatedlatlon["Name"] == image_name]

        # Apply the scaling function to the subset of detections for the current photo
        scaled_detections = scale_first_boxnew(detections_subset, photo_subset, interpolate_x_bounds, interpolate_y_bounds)

        # Append the scaled detections to the list
        all_scaled_detections.append(scaled_detections)

    # Concatenate all the scaled detections into one DataFrame
    scaled_detections_df = pd.concat(all_scaled_detections, ignore_index=True)

    return scaled_detections_df

def create_boxes_scaledfinal(df, name=None):
    # Ensure the "Name" column exists before filtering
    if name is not None and "Name" in df.columns:
        df = df[df["Name"] == name]

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

    # Ensure "Name" column is added if present in the original dataframe
    if "Name" in df.columns:
        scaled_df.insert(0, "Name", df["Name"].values)

    return scaled_df

def plot_oriented_bboxes(scaled_df, image_name):
    # Filter for the selected image
    df_filtered = scaled_df[scaled_df["Name"] == image_name]

    # Initialize plot
    fig, ax = plt.subplots(figsize=(8, 8))

    for _, row in df_filtered.iterrows():
        x_coords = list(row["x"])
        y_coords = list(row["y"])

        # Close the bounding box polygon by repeating the first point
        ax.plot(x_coords + [x_coords[0]], y_coords + [y_coords[0]], 'r-', linewidth=2)

    # Formatting
    ax.set_title(f"Oriented Bounding Boxes for {image_name}")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_aspect('equal')  # Keep aspect ratio square
    plt.grid(True)
    plt.show()

def plot_multiple_orientedbboxes(scaled_df):
    # Iterate over each unique image name
    for image_name in scaled_df["Name"].unique():
        plot_oriented_bboxes(scaled_df, image_name)

def rotate_bounding_boxes(scaled_df, firstphoto):
    
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
    if "Name" in scaled_df.columns:
        rotated_df.insert(0, "Name", scaled_df["Name"].values)

    return rotated_df

def rotate_bboxes_for_multiple_photos(scaled_df, all_corners_df):
    rotated_bboxes_list = []
    
    for image_name in scaled_df["Name"].unique():
        firstphoto = all_corners_df[all_corners_df["Name"] == image_name]
        
        if firstphoto.empty:
            print(f"Warning: No data found in all_corners_df for image {image_name}. Skipping rotation.")
            continue
        
        scaled_df_filtered = scaled_df[scaled_df["Name"] == image_name]
        
        rotated_bboxes = rotate_bounding_boxes(scaled_df_filtered, firstphoto)
        
        rotated_bboxes_list.append(rotated_bboxes)
    
    final_rotated_df = pd.concat(rotated_bboxes_list, ignore_index=True)
    
    return final_rotated_df

def plot_rotated_bounding_boxes_for_multiple_photos(rotated_df):
    # Iterate over each unique image name
    for image_name in rotated_df["Name"].unique():
        # Filter the rotated dataframe for the current image name
        df_filtered = rotated_df[rotated_df["Name"] == image_name]
        
        # Initialize plot
        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot each rotated bounding box
        for _, row in df_filtered.iterrows():
            x_coords = row["xrot"] + [row["xrot"][0]]  # Close the bounding box
            y_coords = row["yrot"] + [row["yrot"][0]]
            ax.plot(x_coords, y_coords, marker='o', linestyle='-', label=row.get("Name", "Box"))

        # Formatting
        ax.set_xlabel("X (Rotated)")
        ax.set_ylabel("Y (Rotated)")
        ax.set_title(f"Rotated Bounding Boxes for {image_name}")
        ax.legend()
        ax.grid(True)
        plt.show()

def plot_photo_extent_and_detections_for_multiple_photos(corners_df, rotated_df):
    # Rotate all corners for all photos
    rotated_corners_df = rotate_all_corners(corners_df)
    
    # Iterate over each unique image name in the rotated_df dataframe
    for image_name in rotated_df["Name"].unique():
        # Filter the rotated data for the current image name
        df_filtered = rotated_df[rotated_df["Name"] == image_name]
        rotated_corners_filtered = rotated_corners_df[rotated_corners_df["Name"] == image_name]
        
        # Initialize plot
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Exclude the fifth row (centre point) from the rotated photo extent
        x_extent = rotated_corners_filtered["xrot"].iloc[:4].tolist() + [rotated_corners_filtered["xrot"].iloc[0]]  # Close the rectangle
        y_extent = rotated_corners_filtered["yrot"].iloc[:4].tolist() + [rotated_corners_filtered["yrot"].iloc[0]]
        ax.plot(x_extent, y_extent, marker='o', linestyle='--', color="black", label="Photo Extent")

        # Plot rotated bounding boxes (detections) for the current photo
        for _, row in df_filtered.iterrows():
            x_coords = row["xrot"] + [row["xrot"][0]]  # Close the bounding box
            y_coords = row["yrot"] + [row["yrot"][0]]
            ax.plot(x_coords, y_coords, marker='o', linestyle='-', label=row.get("Name", "Box"))

        # Plot the centre as a distinct point (red)
        centre = rotated_corners_filtered.iloc[4]  # Assuming the 5th row is the centre point
        ax.plot(centre["xrot"], centre["yrot"], marker='o', color='red', markersize=8, label="Centre")

        # Formatting
        ax.set_xlabel("X (Rotated)")
        ax.set_ylabel("Y (Rotated)")
        ax.set_title(f"Rotated Photo Extent and Bounding Boxes for {image_name}")
        #ax.legend()
        ax.grid(True)
        plt.show()

def plot_photo_extent_and_detections_for_multiple_photos(corners_df, rotated_df):
    # Rotate all corners for all photos
    rotated_corners_df = rotate_all_corners(corners_df)
    
    # Iterate over each unique image name in the rotated_df dataframe
    for image_name in rotated_df["Name"].unique():
        # Filter the rotated data for the current image name
        df_filtered = rotated_df[rotated_df["Name"] == image_name]
        rotated_corners_filtered = rotated_corners_df[rotated_corners_df["Name"] == image_name]
        
        # Initialize plot
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Exclude the fifth row (centre point) from the rotated photo extent
        x_extent = rotated_corners_filtered["xrot"].iloc[:4].tolist() + [rotated_corners_filtered["xrot"].iloc[0]]  # Close the rectangle
        y_extent = rotated_corners_filtered["yrot"].iloc[:4].tolist() + [rotated_corners_filtered["yrot"].iloc[0]]
        ax.plot(x_extent, y_extent, marker='o', linestyle='--', color="black", label="Photo Extent")

        # Plot rotated bounding boxes (detections) for the current photo
        for _, row in df_filtered.iterrows():
            x_coords = row["xrot"] + [row["xrot"][0]]  # Close the bounding box
            y_coords = row["yrot"] + [row["yrot"][0]]
            ax.plot(x_coords, y_coords, marker='o', linestyle='-', label=row.get("Name", "Box"))

        # Plot the centre as a distinct point (red)
        centre = rotated_corners_filtered.iloc[4]  # Assuming the 5th row is the centre point
        ax.plot(centre["xrot"], centre["yrot"], marker='o', color='red', markersize=8, label="Centre")

        # Formatting
        ax.set_xlabel("X (Rotated)")
        ax.set_ylabel("Y (Rotated)")
        ax.set_title(f"Rotated Photo Extent and Bounding Boxes for {image_name}")
        #ax.legend()
        ax.grid(True)
        plt.show()

def plot_photo_extent_and_detections_for_multiple_photos(corners_df, rotated_df):
    # Rotate all corners for all photos
    rotated_corners_df = rotate_all_corners(corners_df)
    
    # Iterate over each unique image name in the rotated_df dataframe
    for image_name in rotated_df["Name"].unique():
        # Filter the rotated data for the current image name
        df_filtered = rotated_df[rotated_df["Name"] == image_name]
        rotated_corners_filtered = rotated_corners_df[rotated_corners_df["Name"] == image_name]
        
        # Initialize plot
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Exclude the fifth row (centre point) from the rotated photo extent
        x_extent = rotated_corners_filtered["xrot"].iloc[:4].tolist() + [rotated_corners_filtered["xrot"].iloc[0]]  # Close the rectangle
        y_extent = rotated_corners_filtered["yrot"].iloc[:4].tolist() + [rotated_corners_filtered["yrot"].iloc[0]]
        ax.plot(x_extent, y_extent, marker='o', linestyle='--', color="black", label="Photo Extent")

        # Plot rotated bounding boxes (detections) for the current photo
        for _, row in df_filtered.iterrows():
            x_coords = row["xrot"] + [row["xrot"][0]]  # Close the bounding box
            y_coords = row["yrot"] + [row["yrot"][0]]
            ax.plot(x_coords, y_coords, marker='o', linestyle='-', label=row.get("Name", "Box"))

        # Plot the centre as a distinct point (red)
        centre = rotated_corners_filtered.iloc[4]  # Assuming the 5th row is the centre point
        ax.plot(centre["xrot"], centre["yrot"], marker='o', color='red', markersize=8, label="Centre")

        # Formatting
        ax.set_xlabel("X (Rotated)")
        ax.set_ylabel("Y (Rotated)")
        ax.set_title(f"Rotated Photo Extent and Bounding Boxes for {image_name}")
        #ax.legend()
        ax.grid(True)
        plt.show()

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

def plot_multiple_realworld_extents(rotatedcornerswithcoords, realworldphotos, title="Rotated Photo Extent and Bounding Boxes"):
    fig, ax = plt.subplots(figsize=(8, 8))  # Create a single figure and axis
    
    # Get unique photo names that exist in realworldphotos
    valid_photos = realworldphotos["Name"].unique()

    # Filter rotatedcornerswithcoords to only include valid photos
    filtered_corners = rotatedcornerswithcoords[rotatedcornerswithcoords["Name"].isin(valid_photos)]
    filtered_detections = realworldphotos[realworldphotos["Name"].isin(valid_photos)]
    
    for photo_name in valid_photos:
        # Get data for the current photo
        corners_photo = filtered_corners[filtered_corners["Name"] == photo_name]
        rotated_photo = filtered_detections[filtered_detections["Name"] == photo_name]

        # Ensure there are at least 5 rows (4 corners + center)
        if len(corners_photo) < 5:
            continue  # Skip if data is insufficient

        # Plot photo extent (excluding the center point)
        lat_extent = corners_photo["latrot"].iloc[:4].tolist() + [corners_photo["latrot"].iloc[0]]  
        lon_extent = corners_photo["lonrot"].iloc[:4].tolist() + [corners_photo["lonrot"].iloc[0]]
        ax.plot(lon_extent, lat_extent, marker='o', linestyle='--', color="black", label=f"{photo_name} Extent")

        # Plot bounding boxes in lat/lon
        for _, row in rotated_photo.iterrows():
            if isinstance(row["latrot"], list) and isinstance(row["lonrot"], list):
                lat_coords = row["latrot"] + [row["latrot"][0]]  # Close the bounding box
                lon_coords = row["lonrot"] + [row["lonrot"][0]]
            else:
                lat_coords = np.array(row["latrot"]).tolist() + [row["latrot"][0]]  
                lon_coords = np.array(row["lonrot"]).tolist() + [row["lonrot"][0]]

            ax.plot(lon_coords, lat_coords, marker='o', linestyle='-')

        # Plot centre as a red point
        centre = corners_photo.iloc[4]  # Assuming the 5th row is the centre point
        ax.plot(centre["lonrot"], centre["latrot"], marker='o', color='red', markersize=8)
    
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    
    plt.show()