import os
import pandas as pd
from functions.videofunctions import *
from functions.mapfunctions import plot_video_frame_detections_map
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=""
    )

    parser.add_argument(
        "--flight_extract_folder",
        type=str,
        required=True
    )

    parser.add_argument(
        "--full_flight_metadata",
        type=str,
        required=True
    )

    parser.add_argument(
        "--videoresults_folder",
        type=str,
        required=True
    )

    parser.add_argument(
        "--qgistiff_folder",
        type=str,
        required=True
    )

    parser.add_argument(
        "--finalmaps_folder",
        type=str,
        required=True
    )

    parser.add_argument(
        "--flight_record_metadata_folder",
        type=str,
        required=True
    )

    parser.add_argument(
        "--bbcount_folder",
        type=str,
        required=True
    )

    return parser.parse_args()

def main():
    import time
    from datetime import datetime
    
    args = parse_arguments()

    # Paths
    # output of A
    flight_extract_folder = args.flight_extract_folder
    # output of A
    full_flight_metadata = args.full_flight_metadata
    # output of B
    videoresults_folder = args.videoresults_folder
    # tif files of satelite images extracted from qgis for flight extents of each unique flight 
    # can be automated but in the future
    qgistiff_folder = args.qgistiff_folder
    # output 1 - maps with OBBs
    finalmaps_folder = args.finalmaps_folder
    # output of C
    flight_record_metadata_folder = args.flight_record_metadata_folder
    # output 2 - bounding boxes
    bbcount_folder = args.bbcount_folder
    os.makedirs(bbcount_folder, exist_ok=True)  
    os.makedirs(finalmaps_folder, exist_ok=True)

    # Load metadata that links videos to flight records
    videometadata = pd.read_csv(full_flight_metadata)

    timing_records = []

    # Process each unique flight record
    # Videos are nested within each flight record
    for flight_record_name in videometadata['FlightRecord'].unique():

        start_time = time.perf_counter()
        start_timestamp = datetime.utcnow().isoformat()

        flight_record_path = os.path.join(flight_record_metadata_folder, flight_record_name)
        if not os.path.exists(flight_record_path):
            print(f"Flight record file {flight_record_name} not found, skipping.")
            continue

        print(f"\nProcessing flight record: {flight_record_name}")

        # Read full flight record CSV
        videocsv = read_flight_csv(flight_record_path)

        # Get all videos associated with this flight record
        flight_videos = videometadata[videometadata['FlightRecord'] == flight_record_name]['FileName'].tolist()

        all_group_rows = []

        # Video processing
        for videoname in flight_videos:
            videoname_noext = os.path.splitext(videoname)[0]
            extract_path = os.path.join(flight_extract_folder, f"{videoname_noext}_extract.csv")

            if not os.path.exists(extract_path):
                print(f"Extract file for {videoname} not found, skipping.")
                continue

            group_rows = pd.read_csv(extract_path)
            group_rows['videoname'] = videoname
            group_rows['group'] = 1

            # solve trigo for frame extent
            group_rows = extract_datetime_components(group_rows, datetime_col="datetime(utc)")
            group_rows = adjust_time(group_rows)
            group_rows = select_columns(group_rows)
            group_rows = add_pitch_and_rotation(group_rows)
            group_rows = thetaData(group_rows)
            group_rows = convert_ascentData(group_rows)
            group_rows = calculate_h(group_rows)
            group_rows = calculate_o(group_rows)
            group_rows = calculate_thetaphi(group_rows)
            group_rows = calculate_theta_minus_fov_vv(group_rows)
            group_rows = calculate_bdfd(group_rows)
            group_rows = calculate_thetaphi_comp(group_rows)
            group_rows = calculate_bd(group_rows)
            group_rows = calculate_bdfdh(group_rows)
            group_rows = calculate_alt(group_rows)
            group_rows = calculate_width(group_rows)
            group_rows = calculate_a1a2(group_rows)
            group_rows = calculate_hypo(group_rows)
            group_rows = calculate_seh(group_rows)
            group_rows = calculate_db1(group_rows)
            group_rows = calculate_b1b2(group_rows)
            group_rows = add_frame_numbers(group_rows)

            all_group_rows.append(group_rows)

        if not all_group_rows:
            print(f"No valid videos found for {flight_record_name}, skipping.")
            continue

        group_rows_combined = pd.concat(all_group_rows, ignore_index=True)

        # Generate corners and correct orientation - project to lat lon coords
        videocorners = create_all_video_corners(group_rows_combined)
        videocorners = videocorners.rename(columns={"video": "Name"})
        videocorners = rotate_all_video_corners(videocorners)
        rotatedvideocornerswithcoords = merge_video_corners_with_coordinates(videocorners, group_rows_combined)
        allvideocornersrotatedlatlon = calculate_video_rotated_latlon(rotatedvideocornerswithcoords, 1/111320)
        hull_data = get_video_hulls(rotatedvideocornerswithcoords)
        all_realworldvideosfiltered = []

        # Detection processing
        for videoname in flight_videos:
            detection_csv_path = os.path.join(videoresults_folder, f"{os.path.splitext(videoname)[0]}_detections.csv")

            if not os.path.exists(detection_csv_path):
                print(f"Detection results for {videoname} not found, skipping.")
                continue

            detectionsvideo = read_flight_csv(detection_csv_path)
            detectionsvideo["filename"] = detectionsvideo["filename"].str.replace('_640x640', '', regex=False)
            detectionsvideo = detectionsvideo.rename(columns={"filename": "videoname"})

            group_rows_this_video = group_rows_combined[group_rows_combined['videoname'] == videoname]
            detectionsvideo = add_max_frame_number(detectionsvideo, group_rows_this_video)
            detectionsvideo = normalize_and_round_frames(detectionsvideo)
            detectionsvideo = filter_top_confidencev2(detectionsvideo)

            detectionvideoswithdims = merge_video_detections_with_coordinates(detectionsvideo, group_rows_this_video)
            detectionvideoswithdims = flip_y_coordinates(detectionvideoswithdims)

            boxes_scaled = process_detections(detectionvideoswithdims, allvideocornersrotatedlatlon, interpolate_x_boundsnew, interpolate_y_boundsnew)
            boxesscaledfinal = create_boxes_scaledfinal_video(boxes_scaled)

            extracted_rows = extract_videoname_and_frame(boxesscaledfinal)
            frames_corners = get_frame_corners_for_unique_keys(extracted_rows, allvideocornersrotatedlatlon)
            clusters = process_video_frames(frames_corners)
            boxesscaledfinal = assign_clusters(boxesscaledfinal, clusters)
            selecteddetections = extract_frames_with_min_bboxes(boxesscaledfinal)
            extracted_rowsr2 = extract_videoname_and_frame(selecteddetections)
            frames_cornersr2 = get_frame_corners_for_unique_keys(extracted_rowsr2, allvideocornersrotatedlatlon)
            filtered_frames, filtered_extracted_rows = filter_frames_by_overlap(frames_cornersr2, extracted_rowsr2)
            filtered_selecteddetections = filter_selected_detections_by_frames(filtered_extracted_rows, selecteddetections)
            final_rotated_bboxes_videos = rotate_bboxes_for_videos(filtered_selecteddetections, allvideocornersrotatedlatlon)
            realworldvideos = convert_to_real_world_coordinates(final_rotated_bboxes_videos)
            realworldvideosfiltered = filter_by_overlapdetectionsfinal(realworldvideos, overlap_threshold=30)

            all_realworldvideosfiltered.append(realworldvideosfiltered)

        if not all_realworldvideosfiltered:
            raise RuntimeError(f"No valid detections for {flight_record_name}")

        combined_realworld = pd.concat(all_realworldvideosfiltered, ignore_index=True)

        # Export results
        output_csv_path = os.path.join(bbcount_folder, f"{flight_record_name.replace('.csv', '')}_summary.csv")
        export_final_detection_summary(combined_realworld, output_csv_path)

        # Generate map
        tif_path = os.path.join(qgistiff_folder, flight_record_name.replace('.csv', '.tif'))
        if not os.path.exists(tif_path):
            print(f"Geotiff {tif_path} not found, skipping map.")
            continue

        fig = plot_video_frame_detections_map(rotatedvideocornerswithcoords, combined_realworld, tif_path, hull_data, title="Geotiff with Detections")
        plt.show()  # <-- Shows map in Python
        # Save map after interactive view
        output_map_path = os.path.join(finalmaps_folder, f"{flight_record_name.replace('.csv','')}_detections_map.png")
        fig.savefig(output_map_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        elapsed_seconds = time.perf_counter() - start_time

        timing_records.append({
            "FlightRecord": flight_record_name,
            "start_time_utc": start_timestamp,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "num_videos": len(flight_videos),
            "num_detections": len(combined_realworld)
        })

        print(
            f"Finished {flight_record_name} in "
            f"{elapsed_seconds:.2f} seconds "
            f"({len(combined_realworld)} detections)"
        )

    timing_df = pd.DataFrame(timing_records)
    timing_csv_path = os.path.join(
        bbcount_folder, "flight_record_processing_times.csv"
    )
    timing_df.to_csv(timing_csv_path, index=False)

    print(f"\nTiming summary saved to: {timing_csv_path}")

    print("\nFinished processing flight record:", flight_record_name)

if __name__ == "__main__":
    main()
