### Turtle Track Detection

To run the preprocessing, use the following command: (replace the paths with the path of where you downloaded the data or where you stored your personal data)
```
python src/Acleanflightprocessv2.py --video_folder ./data/cleanvideos --flight_folder ./data/allflightdata --output_folder ./data/flightforuniquevideosv3 --output_metadata ./data/flightforuniquevideosv3/video_metadatav3.csv --output_metadata_categorized ./data/flightforuniquevideosv3/video_metadata_categorizedv3.csv --output_frames ./data/flightforuniquevideosv3/rows_extracted_summary.csv
```

To run the turtle track detections on turtle track videos, use the following command: (replace the paths with the path of where you downloaded the data or where you stored your personal data)
```
python src/Bvideomodelrunv2.py --ogvideo_folder ./data/cleanvideos --flight_folder ./data/allflightdata --metadata_categorized ./data/flightforuniquevideosv3/video_metadata_categorizedv3.csv --resized_video_folder ./data/resized_folder --detection_output_folder ./results_video --summary_csv_path ./results_video/video_processing_timesv3.csv --model_path ./data/model/YOLOv8S.pt --output_frame_screencapture_folder ./results_video/duplicatevideoframesv2
```

To run the turtle track detections on turtle track images, use the following command: (replace the paths with the path of where you downloaded the data or where you stored your personal data)
```
python src/Bphotomodelrun.py --model_path ./data/model/YOLOv8S.pt --photos_path ./data/photos --output_detections ./results/detectionsgraph.csv 
```

To run the mapping of the turtle track detections from photos onto a satellite image use the following command: (replace the paths with the path of where you downloaded the data)
```
python src/Dphotosmap.py --airdata_flight_record ./data/photosmetadataforD/19th.csv --qgis_metadata ./data/photosmetadataforD/19thcsv.csv --detections ./results/detectionsgraph.csv --satellite_image ./data/qgistiff/clippedCHTS.tif
```

To get the video metadata, use the following command: (replace the paths with the path of where you downloaded the data or where you stored your personal data)
```
python src/Cvideometadata.py --ogvideo_folder ./data/cleanvideos --metadata_folder ./data/videometadata
```

To run the mapping of the turtle track detections from videos onto a satellite image use the following command: (replace the paths with the path of where you downloaded the data)
```
python src/Dvideomap.py --flight_extract_folder ./data/flightforuniquevideosv3 --full_flight_metadata ./data/flightforuniquevideosv3/video_metadata_categorizedv3.csv --videoresults_folder ./results_video --qgistiff_folder ./data/qgistiff --finalmaps_folder ./results_video/maps --flight_record_metadata_folder ./data/videometadata --bbcount_folder ./results_video/bbcount
```
