import subprocess
import os

# Define directories for output files
VIDEO_DIR = "database_videos"
AUDIO_DIR = "database_audios"

def separate_video_audio(video_path):
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    filename = os.path.basename(video_path).split('.')[0]
    audio_path = os.path.join(AUDIO_DIR, f"{filename}.wav")
    video_path_no_audio = os.path.join(VIDEO_DIR, f"{filename}.mp4")

    # Check if the video contains an audio stream
    probe_command = f'ffprobe -i "{video_path}" -show_streams -select_streams a -loglevel error'
    result = subprocess.run(probe_command, shell=True, capture_output=True, text=True)

    if result.stdout.strip():  # If output is not empty, audio stream exists
        print("Audio stream detected. Extracting audio...")
        audio_command = f'ffmpeg -i "{video_path}" -vn -ar 44100 -ac 2 -acodec pcm_s16le "{audio_path}"'
        subprocess.run(audio_command, shell=True, check=True)
    else:
        print("No audio stream detected. Generating silent .wav file...")
        silent_audio_command = f'ffmpeg -f lavfi -i anullsrc=r=44100:cl=stereo -t 5 "{audio_path}"'
        subprocess.run(silent_audio_command, shell=True, check=True)

    print(f"\n\n separating video:{video_path} \n\n")
    # Extract video without audio
    video_command = f'ffmpeg -i "{video_path}" -c:v copy -an "{video_path_no_audio}"'
    subprocess.run(video_command, shell=True, check=True)

# Example usage
if __name__ == "__main__":
    input_video_path = "C:/Users/KIIT/projects/minor_project_/minor_project/uploaded_videos/1743037434_video_1743037403.mp4"
    separate_video_audio(input_video_path)
