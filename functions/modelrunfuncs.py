import os
import subprocess
import cv2
import time
import csv
import pandas as pd
from ultralytics import YOLO
import numpy as np

# resize to 640 x 640 as that is dimension for training data/computationally feasible
def resize_video(input_path, output_path):
    try:
        subprocess.run([
            "ffmpeg", "-i", input_path, "-vf", "scale=640:640",
            "-c:a", "copy", output_path
        ], check=True)
        print(f"Resized video saved to {output_path}")
    except subprocess.CalledProcessError:
        print(f"Failed to resize {input_path}")


# some videos need to be trimmed as video does not end when flight ends
def trim_video(input_path, output_path, end_time_seconds):
    
    temp_path = output_path + ".tmp.mp4"  # or use .mov depending on your input

    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path, "-t", str(end_time_seconds),
            "-c", "copy", temp_path
        ], check=True)
        os.replace(temp_path, output_path)
        print(f"Trimmed video saved to {output_path}")
    except subprocess.CalledProcessError:
        print(f"Failed to trim {input_path}")
    except Exception as e:
        print(f"Unexpected error during trim: {e}")

# Run YOLO on video with oriented bounding boxes, save annotated frames
def run_yolo_on_videov2(model_path, video_path, output_csv_path, save_frames=False, screencap_dir=None):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(fps * 0.1))

    if save_frames and screencap_dir:
        os.makedirs(screencap_dir, exist_ok=True)

    with open(output_csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "filename", "frame_fraction", "total_frames", "class_id",
            "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "confidence"
        ])

        frame_count = 0
        saved_count = 0
        start_time = time.time()

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            frame_fraction = min(frame_count / total_frames, 1)
            results = model(frame)
            obbs = results[0].obb

            has_detection = False
            frame_copy = None

            for obb in obbs:
                class_id = int(obb.cls)
                confidence = float(obb.conf)
                xyxyxyxyn = obb.xyxyxyxyn

                # Extract points exactly as per the tensor format
                x1, y1 = xyxyxyxyn[0][0][0].item(), xyxyxyxyn[0][0][1].item()
                x2, y2 = xyxyxyxyn[0][1][0].item(), xyxyxyxyn[0][1][1].item()
                x3, y3 = xyxyxyxyn[0][2][0].item(), xyxyxyxyn[0][2][1].item()
                x4, y4 = xyxyxyxyn[0][3][0].item(), xyxyxyxyn[0][3][1].item()

                writer.writerow([
                    os.path.basename(video_path), frame_fraction, total_frames, class_id,
                    x1, y1, x2, y2, x3, y3, x4, y4, confidence
                ])

                if save_frames:
                    if frame_copy is None:
                        frame_copy = frame.copy()

                    h, w = frame.shape[:2]
                    pts = [
                        (int(x1 * w), int(y1 * h)),
                        (int(x2 * w), int(y2 * h)),
                        (int(x3 * w), int(y3 * h)),
                        (int(x4 * w), int(y4 * h))
                    ]

                    cv2.polylines(frame_copy, [np.array(pts)], isClosed=True, color=(0, 255, 0), thickness=2)
                    label = f"{class_id} {confidence:.2f}"
                    cv2.putText(frame_copy, label, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    has_detection = True

            if save_frames and has_detection and frame_copy is not None:
                frame_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame{frame_count}.jpg"
                save_path = os.path.join(screencap_dir, frame_name)
                cv2.imwrite(save_path, frame_copy)
                saved_count += 1

        cap.release()
        processing_time = round(time.time() - start_time, 2)
        print(f"Processed {os.path.basename(video_path)} in {processing_time} seconds. Results saved to {output_csv_path}")
        if save_frames:
            print(f"{saved_count} annotated frames saved to {screencap_dir}")


def load_metadatav2(metadata_path, flight_folder):
    df_metadata = pd.read_csv(metadata_path)
    df_metadata['NeedsTrim'] = df_metadata['NeedsTrim'].astype(bool)
    base_names = df_metadata['FileName'].str.rsplit('.', n=1).str.get(0)

    needs_trim_dict = dict(zip(base_names, df_metadata['NeedsTrim']))
    end_time_dict = dict(zip(base_names, df_metadata['EndDate']))
    start_time_dict = dict(zip(base_names, df_metadata['CreateDate_local']))
    flight_record_dict = dict(zip(base_names, df_metadata['FlightRecord']))

    flight_time_bounds = {}
    for flight_file in os.listdir(flight_folder):
        if not flight_file.endswith('.csv'):
            continue
        flight_path = os.path.join(flight_folder, flight_file)
        df_flight = pd.read_csv(flight_path)
        if 'datetime_local' not in df_flight.columns:
            continue
        df_flight['datetime_local'] = pd.to_datetime(df_flight['datetime_local'])
        flight_time_bounds[flight_file] = (df_flight['datetime_local'].iloc[0], df_flight['datetime_local'].iloc[-1])

    return needs_trim_dict, end_time_dict, start_time_dict, flight_record_dict, flight_time_bounds


def process_videov2(video_file, paths, metadata_dicts, model_path, save_frames=False):
    og_folder = paths['og']
    resized_folder = paths['resized']
    detection_folder = paths['detection']
    screencap_folder = paths.get('screencap', None)

    needs_trim_dict, end_time_dict, start_time_dict, flight_record_dict, flight_time_bounds = metadata_dicts

    base_name = os.path.splitext(video_file)[0]
    if base_name not in needs_trim_dict:
        print(f"Skipping {video_file} (no metadata found).")
        return None

    original_path = os.path.join(og_folder, video_file)
    resized_path = os.path.join(resized_folder, f"{base_name}_640x640{os.path.splitext(video_file)[1]}")
    detection_csv_path = os.path.join(detection_folder, f"{base_name}_detections.csv")

    resize_start = time.time()
    if not os.path.exists(resized_path):
        resize_video(original_path, resized_path)
    else:
        print(f"Resized copy already exists for {video_file}, skipping resizing.")
    resize_duration = round(time.time() - resize_start, 2)

    if needs_trim_dict[base_name]:
        print(f"Trimming resized copy of {video_file}...")
        video_start = pd.to_datetime(start_time_dict[base_name]).round('S')
        video_end = pd.to_datetime(end_time_dict[base_name]).round('S')
        flight_record_name = flight_record_dict[base_name]

        if flight_record_name not in flight_time_bounds:
            print(f"Flight record for {video_file} not found, skipping trim.")
        else:
            flight_start, flight_end = flight_time_bounds[flight_record_name]
            actual_trim_end = min(video_end, flight_end)
            duration_seconds = (actual_trim_end - video_start).total_seconds()

            if duration_seconds <= 0:
                print(f"Warning: Computed trim duration for {video_file} is <= 0 seconds. Skipping trim.")
            else:
                trim_video(resized_path, resized_path, duration_seconds)

    detection_start = time.time()
    if not os.path.exists(detection_csv_path):
        run_yolo_on_videov2(
            model_path=model_path,
            video_path=resized_path,
            output_csv_path=detection_csv_path,
            save_frames=save_frames,
            screencap_dir=screencap_folder
        )
    else:
        print(f"Detection results already exist for {video_file}, skipping detection.")
    detection_duration = round(time.time() - detection_start, 2)

    total_duration = round(resize_duration + detection_duration, 2)

    return {
        'VideoName': video_file,
        'ResizingTime_s': resize_duration,
        'YOLODetectionTime_s': detection_duration,
        'TotalTime_s': total_duration
    }