import ffmpeg
from tqdm import tqdm
import os


def convert_videos_in_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".webm"):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + ".mp4"
            output_path = os.path.join(output_folder, output_filename)

            # Get original video information
            probe = ffmpeg.probe(input_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            if not video_stream:
                print(f"⚠️ {filename}: Video stream not found. Skipping.")
                continue

            original_width = int(video_stream['width'])  # Original width of the video
            original_height = int(video_stream['height'])  # Original height of the video

            # Adjust width to the nearest even number
            new_width = original_width - (original_width % 2)

            # Execute FFmpeg conversion
            (
                ffmpeg
                .input(input_path)
                .output(output_path, vcodec="libx264", acodec="aac", vf=f"scale={new_width}:{original_height}")
                .run(overwrite_output=True, quiet=True)
            )

    print("✅ All conversions completed!")

# Set folders for conversion
input_folder = "20bn-something-something-v2"  # Folder containing original webm files
output_folder = "20bn-something-something-v2-mp4"  # Folder where converted MP4 files will be saved

convert_videos_in_folder(input_folder, output_folder)
