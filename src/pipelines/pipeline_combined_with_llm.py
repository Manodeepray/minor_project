from .core.record_face import record_class    
from .core.attendance import get_attendance_from_video
from .core.tools.attendance_to_csv import save_dicts_to_csv
from .core.tools.convert_to_video import frames_to_video
from .core.analysis import analyze_attendance_csv 
from .core.conv_vid_format import convert_to_h264
import os

from .core.llm_integration import record_audio,transcribe_audio
import threading


# Define directories and recording time
AUDIO_DIR = "./recorded_audio"
RECORDED_VIDEO_DIR = "recorded_videos"
RECORD_TIME = 5
TRANSCRIPT_DIR = "./transcription"

# Function to record video
# Shared dictionary to store results from threads
results = {}

# Function to record video
def record_video(results):
    video_file_path = record_class(output_dir=RECORDED_VIDEO_DIR)
    results["video_path"] = video_file_path
    print(f"Video recorded and saved at: {video_file_path}")

# Function to record audio
def record_audio_parallel(results):
    output_audio_path = record_audio(AUDIO_DIR=AUDIO_DIR, record_seconds=RECORD_TIME)
    results["audio_path"] = output_audio_path
    print(f"Audio recorded and saved at: {output_audio_path}")

# Create threads for video and audio recording
video_thread = threading.Thread(target=record_video, args=(results,))
audio_thread = threading.Thread(target=record_audio_parallel, args=(results,))

# Start the threads
video_thread.start()
audio_thread.start()

# Wait for both threads to complete
video_thread.join()
audio_thread.join()

print("Both video and audio recording completed.")

# Retrieve paths from the results dictionary
video_file_path = results.get("video_path")
output_audio_path = results.get("audio_path")
#### attendance and attentiveness 

# video_file_path= "recorded_videos/29_01_25.mp4"
Attendance , output_frames_dir = get_attendance_from_video(video_path = video_file_path)
csv_filepath = save_dicts_to_csv(dict_list = Attendance ,csv_filename = "./attendance_csv/attendance.csv" )
output_video_path = frames_to_video(output_frames_dir,"./processed_video/output_video.mp4", fps=1)
converted_output_video_path = convert_to_h264(output_video_path)
attendance_df = analyze_attendance_csv(csv_filepath)   
    
    
    
#### lecture transcription 
output_transcript_path = transcribe_audio(TRANSCRIPT_DIR = TRANSCRIPT_DIR ,
                                          AUDIO_FILENAME = output_audio_path)


