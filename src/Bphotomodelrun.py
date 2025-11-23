import cv2
import os
import random
from ultralytics import YOLO
from PIL import Image
import csv
import numpy as np

model = YOLO("/Users/fooyockhaw/Desktop/YOLOv8S.pt")
# put photos here
path = '/Users/fooyockhaw/Desktop/graphabstrac'
# access YOLO detection output here
csv_file_path = '/Users/fooyockhaw/Desktop/results/detectionsgraph.csv'
csv_headers = ["filename", "class_id", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "confidence"]

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_headers)

results = model(path, stream=True, save=True, project='PROJECT', name='NAME')

for i, r in enumerate(results):
    
    im_bgr = r.plot()  
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  
    r.show()

    original_image_path = r.path  # This is the original image path
    original_filename = os.path.basename(original_image_path)  # Get the original filename
    result_image_path = f"/Users/fooyockhaw/Desktop/results/{original_filename}"

    
    im_rgb.save(result_image_path)

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        for obb in r.obb:
            print(f"Inspecting obb: {obb}")
            class_id = int(obb.cls)  # Class ID of the detected object
            confidence = float(obb.conf)  # Confidence score
            xyxyxyxyn = obb.xyxyxyxyn  # Bounding box (center_x, center_y, width, height)

            # retrieve coordinates of the bounding box relative to photo extent (0 to 1)
            x1, y1 = xyxyxyxyn[0][0][0].item(), xyxyxyxyn[0][0][1].item()  # top-left corner
            x2, y2 = xyxyxyxyn[0][1][0].item(), xyxyxyxyn[0][1][1].item()  # top-right corner
            x3, y3 = xyxyxyxyn[0][2][0].item(), xyxyxyxyn[0][2][1].item()  # bottom-right corner
            x4, y4 = xyxyxyxyn[0][3][0].item(), xyxyxyxyn[0][3][1].item() #Write detection data to the CSV, including the image filename
            writer.writerow([original_filename, class_id, x1, y1, x2, y2, x3, y3, x4, y4, confidence])

print(f"All detections saved to {csv_file_path}")




