import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))



import streamlit as st
import cv2
import time
import pandas as pd
from datetime import datetime
from pipelines.core.record_face import record_class    
from pipelines.core.attendance import get_attendance_from_video
from pipelines.core.tools.attendance_to_csv import save_dicts_to_csv
from pipelines.core.tools.convert_to_video import frames_to_video
from pipelines.core.analysis import analyze_attendance_csv 
from pipelines.core import conv_vid_format
import tempfile
from pipelines.core.tools.send_csv import send_csv_Mongodb

import logging
logging.getLogger('streamlit').setLevel(logging.ERROR)

# Ensure necessary directories exist
os.makedirs("./data/raw/recorded_videos", exist_ok=True)
os.makedirs("./data/raw/processed_video", exist_ok=True)
os.makedirs("./data/database/database_attendance_csv", exist_ok=True)

# Initialize session state variables


def process_video(video_file_path , class_name):
    """Processes the recorded video."""
    
    csv_filename=f"./data/database/database_attendance_csv/{class_name}.csv"
    Attendance, output_frames_dir = get_attendance_from_video(video_path=video_file_path)
    
    csv_filepath = save_dicts_to_csv(dict_list=Attendance, csv_filename = csv_filename)
    output_video_path = frames_to_video(output_frames_dir, "./data/processed/processed_video/output_video.mp4", fps=1)

    attendance_df = analyze_attendance_csv(csv_filepath)  
    attendance_df.to_csv(csv_filepath, index=False)
    return output_video_path, csv_filepath

# Streamlit UI
st.title("📹 Class Video Recording & Processing")



# Streamlit UI
st.sidebar.header("Settings")

st.sidebar.write("Enter Class Name or Topic:")
class_name = st.sidebar.text_input("Class Name / Topic", "")


st.sidebar.write("Select Recording Duration:")

# Custom tick marks
duration = st.sidebar.slider("Recording Duration (seconds)", 
                             min_value=10, 
                             max_value=600, 
                             value=10, 
                             step=20)
# Slider with predefined steps


minutes = duration // 60
seconds = duration % 60

# Display in both formats
if minutes > 0:
    st.sidebar.write(f"Selected Duration: {minutes} min {seconds} sec")
else:
    st.sidebar.write(f"Selected Duration: {seconds} sec")



save_dir = st.sidebar.text_input("Save Directory", "./recorded_videos")

# Ensure save directory exists
os.makedirs(save_dir, exist_ok=True)

# Button to start recording
if st.button("Start Recording"):
    # st.write("Recording started...")
    st.markdown("<h2 style='color:red; font-weight:bold;'>🔴 Recording started...</h2>", unsafe_allow_html=True)

    # Open webcam
    cap = cv2.VideoCapture(0)
    
    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = 20  # Frame rate
    
    # Create a temporary file for video
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    video_file_path = os.path.join(save_dir, f"recording_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_file_path, fourcc, fps, (frame_width, frame_height))
    
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video frame")
            break
        
        out.write(frame)
        
        # Show live video
        
    
    # Release resources
    cap.release()
    out.release()
    
    
    # st.write("Recording complete!")
    st.markdown("<h2 style='color:green; font-weight:bold;'>✅ Recording complete!</h2>", unsafe_allow_html=True)
    st.success(f"Recording saved at {video_file_path}")
    
    output_video_path, csv_filepath = process_video(video_file_path = video_file_path,class_name=class_name)
    
    print(f"output :  processed video {output_video_path} | csv file {csv_filepath}")
    try:
        print(f"converting video format {output_video_path}")
        converted_output_video_path = conv_vid_format.convert_to_h264(output_video_path)
        
    except Exception as e:
        print(f"error converting video format {e}")
    # Show processed video
    st.video(converted_output_video_path)
    # video_html = f"""
    #     <video width="700" height="400" controls>
    #         <source src="{converted_output_video_path}" type="video/mp4">
    #         Your browser does not support the video tag.
    #     </video>
    #     """
    # st.markdown(video_html, unsafe_allow_html=True)
    # Display CSV
    attendance_df = pd.read_csv(csv_filepath)
    
       
        
        
    st.write("### 📊 Attendance Records")
    st.dataframe(attendance_df)
    st.download_button("📥 Download CSV", attendance_df.to_csv(index=False), "attendance.csv", "text/csv")

    import time

    while True:
        try:
            send_csv_Mongodb(csv_filepath)
            st.success("✅ CSV sent to database!")
            break  # Exit the loop if successful
        except Exception as e:
            st.error(f"❌ Error sending CSV: {e}. Retrying in 5 seconds...")
            time.sleep(5)  # Wait before retrying
    
    
    
    