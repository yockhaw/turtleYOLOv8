from functions.photofuncsFinal import *
from functions.mapfunctions import *
import pandas as pd
import math
import numpy as np
import time
start_time = time.time()

# this is flight record from Airdata for which the photos are taken
airdatafp = '/Users/fooyockhaw/Desktop/19th.csv'
# this is all the metadata that is extracted using QGIS by uploading photos 
qgisfp = '/Users/fooyockhaw/Desktop/19thcsv.csv'
airdata = read_flight_csv(airdatafp)
qgis = read_flight_csv(qgisfp)
# only retrieve the rows from the flight record where photos were taken - IsPhoto cannot be fully trusted so if rows are missing - using ExifData will be better, just like for videos
IsPhoto = extractIsPhoto(airdata)
IsPhoto = convert_to_umt8(IsPhoto)
pd.set_option("display.max_columns", None)
qgis = extractqgistime(qgis, 'Time')
# match flight record row to each photo
matched_df = get_matched_rows(qgis, IsPhoto)
qgis = arrange_by_time(qgis)
matched_df = arrange_by_time(matched_df)

columnstoappend = ['Name', 'Altitude', 'Path']
matched_df = append_columns_to_dataframe(matched_df, qgis, columnstoappend)
minphotorows = select_columns(matched_df)
# do calculations for photo extent
minphotorows = add_pitch_and_rotation(minphotorows)
minphotorows = thetaData(minphotorows)
minphotorows = convert_ascentData(minphotorows)
minphotorows = calculate_h(minphotorows)
minphotorows = calculate_o(minphotorows)
minphotorows = calculate_thetaphi(minphotorows)
minphotorows = calculate_theta_minus_fov_vv(minphotorows)
minphotorows = calculate_bdfd(minphotorows)
minphotorows = calculate_thetaphi_comp(minphotorows)
minphotorows = calculate_bd(minphotorows)
minphotorows = calculate_bdfdh(minphotorows)
minphotorows = calculate_alt(minphotorows)
minphotorows = calculate_width(minphotorows)
minphotorows = calculate_a1a2(minphotorows)
minphotorows = calculate_hypo(minphotorows)
minphotorows = calculate_seh(minphotorows)
minphotorows = calculate_db1(minphotorows)
minphotorows = calculate_b1b2(minphotorows)
corners = create_all_corners(minphotorows)
print(corners)

corners = corners.rename(columns={"photo": "Name"})
# based on orientation of drone, rotate photo extent
corners = rotate_all_corners(corners)
# plot_proper_corners(corners) # sanity check

# combine dataframes
rotatedcornerswithcoords = merge_corners_with_coordinates(corners, minphotorows)
#print(rotatedcornerswithcoords) # sanity check

# taking coordinates as the drone location, calculate coordinates for each of the corners
meter_to_deg = 1 / 111320
allcornersrotatedlatlon = calculate_rotated_latlon(rotatedcornerswithcoords, meter_to_deg)
print(allcornersrotatedlatlon) # sanity check

#plot_rotated_cornersUpdated(allcornersrotatedlatlon) # sanity check

# detectionsgraph is from model run 
detectionsp = '/Users/fooyockhaw/Desktop/results/detectionsgraph.csv'
detections = read_flight_csv(detectionsp)
detections = detections.rename(columns={"filename": "Name"})

# detections matched to unique photo
detectionswithdims = merge_detections_with_coordinates(detections, minphotorows)
# make detections orientation correct
detectionswithdims = flip_y_coordinates(detectionswithdims)

# skip photos without detections, scale detections - depending on whether the camera is pointing straight down/angled towards the ground
boxes_scaled = process_detections(detectionswithdims, allcornersrotatedlatlon, interpolate_x_boundsnew, interpolate_y_boundsnew)
print(boxes_scaled) # sanity check

# box for photo extent
boxesscaledfinal = create_boxes_scaledfinal(boxes_scaled)
print(boxesscaledfinal) # sanity check

# plot_multiple_orientedbboxes(boxesscaledfinal) # sanity check
# rotate all bounding boxes
final_rotated_bboxes_df = rotate_bboxes_for_multiple_photos(boxesscaledfinal, allcornersrotatedlatlon)
#print(final_rotated_bboxes_df)
#plot_rotated_bounding_boxes_for_multiple_photos(final_rotated_bboxes_df) # sanity check

# plot all
plot_photo_extent_and_detections_for_multiple_photos(corners, final_rotated_bboxes_df)

realworldphotos = convert_to_real_world_coordinates(final_rotated_bboxes_df)
# print(realworldphotos) # sanity check

end_time = time.time()
print(f"Script execution time: {end_time - start_time:.2f} seconds")

# final plot - can save as png if needed
# tif file is also retrieve from QGIS using foc satellite imagery
plot_multiple_realworld_extentsmap(rotatedcornerswithcoords, realworldphotos, "/Users/fooyockhaw/Desktop/clippedCHTS.tif", title="Geotiff with Detections")

