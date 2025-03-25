import os
import subprocess

# Paths
input_video = "./processed_video/output_video.mp4"
output_video = "./processed_video/output_video_h264.mp4"

# Convert video to H.264 codec
def convert_to_h264(input_path):
    print("converting")
    base, ext = os.path.splitext(input_path)  # Split filename and extension
    output_path = f"{base}_h264.mp4"
    
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-vcodec", "libx264", "-acodec", "aac", output_path
    ]
    print("running subprocess")
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"successfully saved at {output_path}")
    return output_path 

# convert_to_h264(input_video, output_video)
