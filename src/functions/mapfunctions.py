import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from functions.photofuncsFinal import plot_multiple_realworld_extents

def plot_geotiff_wgs84onlymap(file_path, cmap="terrain"):

    with rasterio.open(file_path) as dataset:
        # Get current CRS (Coordinate Reference System)
        original_crs = dataset.crs
        target_crs = "EPSG:4326"  # WGS 84 Lat/Lon

        # If the CRS is already WGS 84, plot directly
        if original_crs.to_string() == target_crs:
            dataset_wgs84 = dataset  # Use the original dataset
        else:
            # Compute the transformation
            transform, width, height = calculate_default_transform(
                dataset.crs, target_crs, dataset.width, dataset.height, *dataset.bounds
            )

            # Create new metadata
            new_meta = dataset.meta.copy()
            new_meta.update({
                "crs": target_crs,
                "transform": transform,
                "width": width,
                "height": height
            })

            # Create a MemoryFile and write reprojected data
            memfile = MemoryFile()
            with memfile.open(**new_meta) as mem_dataset:
                for i in range(1, dataset.count + 1):
                    reproject(
                        source=rasterio.band(dataset, i),
                        destination=rasterio.band(mem_dataset, i),
                        src_transform=dataset.transform,
                        src_crs=dataset.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest
                    )
            
            # Reopen MemoryFile as a reader
            dataset_wgs84 = memfile.open()

        # Plot the final dataset
        fig, ax = plt.subplots(figsize=(10, 6))
        show(dataset_wgs84, ax=ax, cmap=cmap)  # Now dataset_wgs84 is a reader
        ax.set_title("Reprojected GeoTIFF (WGS 84)")

        # Get new bounding box in lat/lon
        left, bottom, right, top = dataset_wgs84.bounds
        ax.set_xlabel(f"Longitude ({left:.2f} to {right:.2f})")
        ax.set_ylabel(f"Latitude ({bottom:.2f} to {top:.2f})")

        plt.show()

        # Close dataset if it was created in memory
        dataset_wgs84.close()
        memfile.close()



def plot_realworldextent_no_legendmap(corners_df, rotated_df, ax, title="Rotated Photo Extent and Bounding Boxes"):

    # Plot photo extent in lat/lon
    lat_extent = corners_df["latrot"].tolist() + [corners_df["latrot"].iloc[0]]  # Close the rectangle
    lon_extent = corners_df["lonrot"].tolist() + [corners_df["lonrot"].iloc[0]]
    ax.plot(lon_extent, lat_extent, marker='o', linestyle='--', color="black")

    # Plot bounding boxes in lat/lon
    for _, row in rotated_df.iterrows():
        if isinstance(row["latrot"], list) and isinstance(row["lonrot"], list):
            lat_coords = row["latrot"] + [row["latrot"][0]]  # Close the bounding box
            lon_coords = row["lonrot"] + [row["lonrot"][0]]
        else:
            lat_coords = np.array(row["latrot"]).tolist() + [row["latrot"][0]]  
            lon_coords = np.array(row["lonrot"]).tolist() + [row["lonrot"][0]]

        ax.plot(lon_coords, lat_coords, marker='o', linestyle='-')

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(True)

import numpy as np

def plot_realworldextent_no_legendmapUpdated(corners_df, rotated_df, ax, title="Rotated Photo Extent and Bounding Boxes"):

    # Exclude the fifth row (centre point) from the photo extent
    lat_extent = corners_df["latrot"].iloc[:4].tolist() + [corners_df["latrot"].iloc[0]]  # Close the rectangle
    lon_extent = corners_df["lonrot"].iloc[:4].tolist() + [corners_df["lonrot"].iloc[0]]
    ax.plot(lon_extent, lat_extent, marker='o', linestyle='--', color="black")

    # Plot bounding boxes in lat/lon
    for _, row in rotated_df.iterrows():
        if isinstance(row["latrot"], list) and isinstance(row["lonrot"], list):
            lat_coords = row["latrot"] + [row["latrot"][0]]  # Close the bounding box
            lon_coords = row["lonrot"] + [row["lonrot"][0]]
        else:
            lat_coords = np.array(row["latrot"]).tolist() + [row["latrot"][0]]  
            lon_coords = np.array(row["lonrot"]).tolist() + [row["lonrot"][0]]

        ax.plot(lon_coords, lat_coords, marker='o', linestyle='-')

    # Plot centre as a red point
    centre = corners_df.iloc[4]  # Assuming the 5th row is the centre point
    ax.plot(centre["lonrot"], centre["latrot"], marker='o', color='red', markersize=8)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(True)




def plot_geotiff_wgs84withdetections(file_path, ax, cmap="terrain"):
    
    with rasterio.open(file_path) as dataset:
        # Get current CRS (Coordinate Reference System)
        original_crs = dataset.crs
        target_crs = "EPSG:4326"  # WGS 84 Lat/Lon

        # If the CRS is already WGS 84, plot directly
        if original_crs.to_string() == target_crs:
            dataset_wgs84 = dataset  # Use the original dataset
        else:
            # Compute the transformation
            transform, width, height = calculate_default_transform(
                dataset.crs, target_crs, dataset.width, dataset.height, *dataset.bounds
            )

            # Create new metadata
            new_meta = dataset.meta.copy()
            new_meta.update({
                "crs": target_crs,
                "transform": transform,
                "width": width,
                "height": height
            })

            # Create a MemoryFile and write reprojected data
            memfile = MemoryFile()
            with memfile.open(**new_meta) as mem_dataset:
                for i in range(1, dataset.count + 1):
                    reproject(
                        source=rasterio.band(dataset, i),
                        destination=rasterio.band(mem_dataset, i),
                        src_transform=dataset.transform,
                        src_crs=dataset.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest
                    )
            
            # Reopen MemoryFile as a reader
            dataset_wgs84 = memfile.open()

        show(dataset_wgs84, ax=ax, cmap=cmap)



def overlay_detections_on_geotiff(geotiff_path, corners_df, rotated_df):

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_geotiff_wgs84withdetections(geotiff_path, ax, cmap="terrain")

    # Overlay the photo extent and bounding boxes
    plot_realworldextent_no_legendmap(corners_df, rotated_df, ax)

    # Display the plot
    plt.show()


def overlay_detections_on_geotiffUpdated(geotiff_path, corners_df, rotated_df):

    unique_photos = corners_df["Name"].unique()
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_geotiff_wgs84withdetections(geotiff_path, ax, cmap="terrain")

    # Overlay the photo extent and bounding boxes
    plot_realworldextent_no_legendmapUpdated(corners_df, rotated_df, ax)

    # Display the plot
    plt.show()


def plot_realworldextent_no_legendmapfinal(corners_df, rotated_df, ax):
    """
    Plots multiple rotated photo extents and bounding boxes on the given axes.
    """
    for photo_name in corners_df["Name"].unique():
        # Filter data for this specific photo
        corners_photo = corners_df[corners_df["Name"] == photo_name]
        rotated_photo = rotated_df[rotated_df["Name"] == photo_name]

        # Exclude the fifth row (centre point) from the photo extent
        lat_extent = corners_photo["latrot"].iloc[:4].tolist() + [corners_photo["latrot"].iloc[0]]
        lon_extent = corners_photo["lonrot"].iloc[:4].tolist() + [corners_photo["lonrot"].iloc[0]]
        ax.plot(lon_extent, lat_extent, marker='o', linestyle='--', color="black")

        # Plot bounding boxes in lat/lon
        for _, row in rotated_photo.iterrows():
            lat_coords = row["latrot"] + [row["latrot"][0]]  # Close the bounding box
            lon_coords = row["lonrot"] + [row["lonrot"][0]]
            ax.plot(lon_coords, lat_coords, marker='o', linestyle='-')

        # Plot centre as a red point
        centre = corners_photo.iloc[4]  # Assuming the 5th row is the centre point
        ax.plot(centre["lonrot"], centre["latrot"], marker='o', color='red', markersize=8)

def overlay_detections_on_geotifffinal(geotiff_path, corners_df, rotated_df):
    """
    Overlays multiple photos' detections onto a single GeoTIFF map.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Load and plot the GeoTIFF
    plot_geotiff_wgs84withdetections(geotiff_path, ax, cmap="terrain")

    # Overlay the detections for all photos
    plot_multiple_realworld_extents(corners_df, rotated_df, ax)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Overlay of All Photos and Detections")
    ax.grid(True)

    plt.show()

def plot_multiple_realworld_extentsmap(rotatedcornerswithcoords, realworldphotos, geotiff_path, title="Rotated Photo Extent and Bounding Boxes"):
    fig, ax = plt.subplots(figsize=(24, 4))
    
    plot_geotiff_wgs84withdetections(geotiff_path, ax)
    
    valid_photos = realworldphotos["Name"].unique()
    filtered_corners = rotatedcornerswithcoords[rotatedcornerswithcoords["Name"].isin(valid_photos)]
    filtered_detections = realworldphotos[realworldphotos["Name"].isin(valid_photos)]
    
    for photo_name in valid_photos:
        corners_photo = filtered_corners[filtered_corners["Name"] == photo_name]
        rotated_photo = filtered_detections[filtered_detections["Name"] == photo_name]

        if len(corners_photo) < 5:
            continue

        lat_extent = corners_photo["latrot"].iloc[:4].tolist() + [corners_photo["latrot"].iloc[0]]  
        lon_extent = corners_photo["lonrot"].iloc[:4].tolist() + [corners_photo["lonrot"].iloc[0]]
        ax.plot(lon_extent, lat_extent, marker='o', linestyle='--', color="black", label=f"{photo_name} Extent")

        for _, row in rotated_photo.iterrows():
            if isinstance(row["latrot"], list) and isinstance(row["lonrot"], list):
                lat_coords = row["latrot"] + [row["latrot"][0]]
                lon_coords = row["lonrot"] + [row["lonrot"][0]]
            else:
                lat_coords = np.array(row["latrot"]).tolist() + [row["latrot"][0]]  
                lon_coords = np.array(row["lonrot"]).tolist() + [row["lonrot"][0]]

            ax.plot(lon_coords, lat_coords, marker='o', linestyle='-')

        centre = corners_photo.iloc[4]
        ax.plot(centre["lonrot"], centre["latrot"], marker='o', color='red', markersize=8)
    
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    
    plt.show()

def plot_multiple_realworld_extentsmap_zoomed(rotatedcornerswithcoords, realworldphotos, geotiff_path,
                                               title=""):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    plot_geotiff_wgs84withdetections(geotiff_path, ax)
    
    valid_photos = realworldphotos["Name"].unique()
    filtered_corners = rotatedcornerswithcoords[rotatedcornerswithcoords["Name"].isin(valid_photos)]
    filtered_detections = realworldphotos[realworldphotos["Name"].isin(valid_photos)]
    
    for photo_name in valid_photos:
        corners_photo = filtered_corners[filtered_corners["Name"] == photo_name]
        rotated_photo = filtered_detections[filtered_detections["Name"] == photo_name]

        if len(corners_photo) < 5:
            continue

        lat_extent = corners_photo["latrot"].iloc[:4].tolist() + [corners_photo["latrot"].iloc[0]]  
        lon_extent = corners_photo["lonrot"].iloc[:4].tolist() + [corners_photo["lonrot"].iloc[0]]
        ax.plot(lon_extent, lat_extent, marker='o', linestyle='--', color="black", label=f"{photo_name} Extent")

        for _, row in rotated_photo.iterrows():
            if isinstance(row["latrot"], list) and isinstance(row["lonrot"], list):
                lat_coords = row["latrot"] + [row["latrot"][0]]
                lon_coords = row["lonrot"] + [row["lonrot"][0]]
            else:
                lat_coords = np.array(row["latrot"]).tolist() + [row["latrot"][0]]  
                lon_coords = np.array(row["lonrot"]).tolist() + [row["lonrot"][0]]

            ax.plot(lon_coords, lat_coords, marker='o', linestyle='-')

        centre = corners_photo.iloc[4]
        ax.plot(centre["lonrot"], centre["latrot"], marker='o', color='red', markersize=8)

    ax.set_xlim(103.00750, 103.00825)
    ax.set_ylim(5.8124, 5.8130)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(True)
    #ax.legend()
    
    plt.show()

def plot_video_hulls(hull_data, ax):
    
    for video_name, hull_points in hull_data.items():
        lon_hull = hull_points[:, 0]
        lat_hull = hull_points[:, 1]
        
        ax.plot(lon_hull, lat_hull, linestyle='-', marker='o', label=f"Hull: {video_name}")

def plot_video_frame_detections_map(rotatedcornerswithcoords, realworldframes, geotiff_path, hull_data, title="Video Frame Detections"):
    fig, ax = plt.subplots(figsize=(24, 4))

    # Plot GeoTIFF as the background
    plot_geotiff_wgs84withdetections(geotiff_path, ax)

    # Get unique (videoname, frame_number) pairs
    unique_video_frames = realworldframes[["videoname", "frame_number"]].drop_duplicates()

    for _, (videoname, frame_number) in unique_video_frames.iterrows():
        # Get detections (bounding boxes) for the current video frame
        rotated_detections = realworldframes[
            (realworldframes["videoname"] == videoname) & 
            (realworldframes["frame_number"] == frame_number)
        ]

        # Plot bounding boxes in lat/lon
        for i, row in rotated_detections.iterrows():
            if isinstance(row["latrot"], list) and isinstance(row["lonrot"], list):
                lat_coords = row["latrot"] + [row["latrot"][0]]  # Close the bounding box
                lon_coords = row["lonrot"] + [row["lonrot"][0]]
            else:
                lat_coords = np.array(row["latrot"]).tolist() + [row["latrot"][0]]  
                lon_coords = np.array(row["lonrot"]).tolist() + [row["lonrot"][0]]

            ax.plot(lon_coords, lat_coords, marker='o', linestyle='-', 
                    label=f"{videoname} - Frame {frame_number}" if i == 0 else "")

    # Plot convex hulls on the same plot
    plot_video_hulls(hull_data, ax)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    
    return fig