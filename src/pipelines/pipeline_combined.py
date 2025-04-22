from .core.record_face import record_class    
from .core.attendance import get_attendance_from_video
from .core.tools.attendance_to_csv import save_dicts_to_csv
from .core.tools.convert_to_video import frames_to_video
from .core.analysis import analyze_attendance_csv 
from .core.conv_vid_format import convert_to_h264
import os
from .core.llm_integration import record_audio,transcribe_audio
from concurrent.futures import ThreadPoolExecutor


AUDIO_DIR = "./data/raw/recorded_audio"
TRANSCRIPT_DIR = "./data/raw/transcription"
RECORD_TIME = 5
RECORDED_VIDEO_DIR = "data/raw/recorded_videos"





video_file_path = record_class(output_dir = RECORDED_VIDEO_DIR) # output_dir = "recorded_videos"

output_audio_path = record_audio(AUDIO_DIR=AUDIO_DIR, record_seconds=RECORD_TIME)


#### attendance and attentiveness 

# video_file_path= "recorded_videos/29_01_25.mp4"
Attendance , output_frames_dir = get_attendance_from_video(video_path = video_file_path)
csv_filepath = save_dicts_to_csv(dict_list = Attendance ,csv_filename = "./data/database/database_attendance_csv/attendance.csv" )
output_video_path = frames_to_video(output_frames_dir,"./data/processed/processed_video/output_video.mp4", fps=1)
converted_output_video_path = convert_to_h264(output_video_path)
attendance_df = analyze_attendance_csv(csv_filepath)   
    
    
    
#### lecture transcription 
output_transcript_path = transcribe_audio(TRANSCRIPT_DIR = TRANSCRIPT_DIR ,
                                          AUDIO_FILENAME = output_audio_path)


