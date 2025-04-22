import os
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from src.pipelines import pipeline_attendance
from src.pipelines.core import llm_integration
from pydub import AudioSegment
from src.pipelines.core.tools import logging_own



VIDEO_DIR = "./data/database/database_videos"
AUDIO_DIR = "./data/database/database_audios"
TRANSCRIPT_DIR = './data/raw/transcription'
NOTES_DIR = "./data/database/database_notes"

# Queue for processing
video_queue = []
lock = threading.Lock()






def convert_mp3_to_wav(mp3_path):
    """Converts an MP3 file to WAV format."""
    try:
        print(f"🎵 Converting {mp3_path} to WAV format...")
        wav_path = mp3_path.replace('.mp3', '.wav')
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")
        print(f"✅ Conversion complete: {wav_path}")
        return wav_path
    except Exception as e:
        print(f"❌ Error converting MP3 to WAV: {e}")
        return None



class VideoHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(('.mp4', '.avi', '.mov')):
            with lock:
                print(f"New video detected: {event.src_path}")
                video_queue.append(event.src_path)







def process_video_and_audio(video_path):
    
    
    print(f"🚀 Processing video: {video_path}")
    
    
    
    video_path = video_path.replace("\r","/r")
    class_name = os.path.splitext(os.path.basename(video_path))[0]  # Extract class name from filename
    print(f"\n class name {class_name}\n")
    
    
    
    output_video_path, csv_filepath, converted_output_video_path = pipeline_attendance.process_video(video_path, class_name)
    
    print(f"✅ Completed video processing: {video_path}")

    # 🔎 Check for a matching audio file
    audio_path = os.path.join(AUDIO_DIR, f"{class_name}.wav")
    if os.path.exists(audio_path):
        pass
    else:
        audio_path = os.path.join(AUDIO_DIR, f"{class_name}.mp3")
    
    audio_path = audio_path.replace("\r","/r")
    print(f"audio path: {audio_path}")
    
    
    
    if ".mp3" in audio_path :
        audio_path = convert_mp3_to_wav(audio_path)
        
    
    if os.path.exists(audio_path):
        print(f"🎯 Found corresponding audio file: {audio_path}")
        process_audio(audio_path,class_name)
    else:
        print(f"❌ No matching audio file found for: {video_path}")

    logging_own.save_processed_video(video_path)









def process_audio(audio_path,class_name):
    try:
        print(f"🎧 Processing audio: {audio_path}")
        
        # ✅ LLM Integration for transcription
        output_transcript_path, transcription = llm_integration.transcribe_audio(
            TRANSCRIPT_DIR=TRANSCRIPT_DIR,
            AUDIO_FILENAME=audio_path ,
            VOSK_MODEL_PATH="models/vosk-model-small-en-us-0.15"
        )
        
        print(f"✅ Transcription saved at: {output_transcript_path}")
        print(f"📝 Transcription:\n{transcription}")
        
        client = llm_integration.load_llm()


        # class_name = os.path.splitext(os.path.basename(output_transcript_path))[0]
        # Generate class notes
        print("generating notes....")
        notes, notes_path = llm_integration.generate_notes(transcription, client, NOTES_DIR, class_name)
        
        print(f"✅ Notes saved at: {notes_path}")
        
        
        
    except Exception as e:
        
        
        print(f"❌ Error processing audio: {e}")





def worker():
    while True:
        with lock:
            if video_queue:
                
                
                video_path = video_queue.pop(0)
                process_video_and_audio(video_path)
        time.sleep(1)

def process_existing_videos():
    """Process all existing videos before starting live monitoring."""
    print("🔍 Checking for existing videos...")

    processed_videos = logging_own.load_processed_videos()
    

    
    
    
    for filename in os.listdir(VIDEO_DIR):
        if filename.endswith(('.mp4', '.avi', '.mov')):
            
            video_path = os.path.join(VIDEO_DIR, filename)
            
            if video_path not in processed_videos:
                with lock:
                    video_queue.append(video_path)
            else:
                print(f"Skipping: {video_path} (Already processed)")
                
                
                
    print(f"✅ Found {len(video_queue)} videos to process.")
    # op = input("continue?yes[y]/no[n]")
    op = "y"
    if op.lower() == 'y':
        pass
    else:
        exit()

def start_monitoring():
    process_existing_videos()  # Process existing videos first

    event_handler = VideoHandler()
    observer = Observer()
    observer.schedule(event_handler, VIDEO_DIR, recursive=False)
    observer.start()

    print("🚀 Monitoring directory for new videos...")

    # Start processing thread
    threading.Thread(target=worker, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_monitoring()
