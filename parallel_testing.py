from concurrent.futures import ThreadPoolExecutor
from record_face import record_class
from llm_integration import record_audio

# Constants
AUDIO_DIR = "./recorded_audio"
RECORD_TIME = 5
RECORDED_VIDEO_DIR = "recorded_videos"

# Function to run record_class and record_audio in parallel
def run_in_parallel():
    with ThreadPoolExecutor() as executor:
        # Submit both tasks to the thread pool
        future_video = executor.submit(record_class, output_dir=RECORDED_VIDEO_DIR)
        future_audio = executor.submit(record_audio, AUDIO_DIR=AUDIO_DIR, record_seconds=RECORD_TIME)
        
        # Wait for both tasks to complete and get their results
        video_file_path = future_video.result()
        output_audio_path = future_audio.result()
    
    return video_file_path, output_audio_path

# Run the functions in parallel
video_file_path, output_audio_path = run_in_parallel()

# Print the results
print(f"Video saved at: {video_file_path}")
print(f"Audio saved at: {output_audio_path}")