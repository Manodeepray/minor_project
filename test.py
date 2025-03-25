import os

video_path = "database_audios/recording_20250311-175857.mp3"
print(os.path.splitext(os.path.basename(video_path))[0])
