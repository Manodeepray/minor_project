import streamlit as st
import os
from pathlib import Path

# Directory paths
VIDEO_DIR = "database_videos"
CSV_DIR = "database_attendance"
NOTES_DIR = "database_notes"
AUDIO_DIR = "database_audios"

def get_files(directory, extensions):
    """Get files with specific extensions from a directory"""
    files = []
    for item in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, item)):
            if any(item.lower().endswith(ext) for ext in extensions):
                files.append(item)
    return files

def display_file_content(file_path):
    """Display content based on file type"""
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.csv':
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    
    elif file_extension == '.mp4':
        st.video(file_path)
    
    elif file_extension in ('.md', '.txt'):
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            st.markdown(content)
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    elif file_extension in ('.mp3', '.wav'):
        st.audio(file_path)
    
    else:
        st.warning(f"Preview not available for {file_extension} files")

def main():
    st.title("Database File Explorer")
    
    # Create tabs for each directory
    tab1, tab2, tab3, tab4 = st.tabs(["Videos", "Attendance", "Notes", "Audios"])
    
    with tab1:
        st.header("Video Files")
        video_files = get_files(VIDEO_DIR, ['.mp4'])
        if video_files:
            selected_video = st.selectbox("Select a video file:", video_files)
            video_path = os.path.join(VIDEO_DIR, selected_video)
            display_file_content(video_path)
        else:
            st.warning("No video files found in the directory")
    
    with tab2:
        st.header("Attendance Records")
        csv_files = get_files(CSV_DIR, ['.csv'])
        if csv_files:
            selected_csv = st.selectbox("Select an attendance file:", csv_files)
            csv_path = os.path.join(CSV_DIR, selected_csv)
            display_file_content(csv_path)
        else:
            st.warning("No CSV files found in the directory")
    
    with tab3:
        st.header("Notes")
        note_files = get_files(NOTES_DIR, ['.md', '.txt'])
        if note_files:
            selected_note = st.selectbox("Select a note file:", note_files)
            note_path = os.path.join(NOTES_DIR, selected_note)
            display_file_content(note_path)
        else:
            st.warning("No note files found in the directory")
    
    with tab4:
        st.header("Audio Files")
        audio_files = get_files(AUDIO_DIR, ['.mp3', '.wav'])
        if audio_files:
            selected_audio = st.selectbox("Select an audio file:", audio_files)
            audio_path = os.path.join(AUDIO_DIR, selected_audio)
            display_file_content(audio_path)
        else:
            st.warning("No audio files found in the directory")

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(NOTES_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    
    main()