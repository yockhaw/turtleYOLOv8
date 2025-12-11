import exiftool
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=""
    )

    parser.add_argument(
        "--ogvideo_folder",
        type=str,
        required=True
    )

    parser.add_argument(
        "--metadata_folder",
        type=str,
        required=True
    )

    return parser.parse_args()

def main():
    args = parse_arguments()
    folder_path = args.ogvideo_folder
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.mpg', '.mpeg', '.3gp', '.wmv'}

    # Create metadata subfolder
    metadata_folder = args.metadata_folder
    os.makedirs(metadata_folder, exist_ok=True)

    video_files = []
    for file in os.listdir(folder_path):
        if os.path.splitext(file)[1].lower() in video_extensions:
            video_files.append(os.path.join(folder_path, file))

    print(f"Found {len(video_files)} video files")

    for video_path in video_files:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = os.path.join(metadata_folder, f"{base_name}_metadata.txt")
        
        try:
            with exiftool.ExifTool() as et:
                metadata = et.execute(video_path)
                
                with open(output_filename, 'w', encoding='utf-8') as f:
                    f.write(metadata)
                
            print(f"Metadata saved to {output_filename}")
        
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

    print("All metadata files saved in 'metadata' subfolder")

if __name__ == "__main__":
    main()