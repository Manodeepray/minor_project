
from .core.record_face import record_class    
from .core.attendance import get_attendance_from_video
from .core.tools.attendance_to_csv import save_dicts_to_csv
from .core.tools.convert_to_video import frames_to_video
from .core.analysis import analyze_attendance_csv 
from .core import conv_vid_format




def process_video(video_file_path , class_name):
    """Processes the recorded video."""
    
    csv_filename=f"./data/database/database_attendance_csv/{class_name}.csv"
 
    
    Attendance, output_frames_dir = get_attendance_from_video(video_path=video_file_path)
    
    csv_filepath = save_dicts_to_csv(dict_list=Attendance, csv_filename = csv_filename)
    
    output_video_path = frames_to_video(output_frames_dir, f"./data/processed/processed_video/{class_name}.mp4", fps=1)

    attendance_df = analyze_attendance_csv(csv_filepath)
      
    attendance_df.to_csv(csv_filepath, index=False)
    
    print(f"output :  processed video {output_video_path} | csv file {csv_filepath}")
    try:
        print(f"converting video format {output_video_path}")
        
        
        converted_output_video_path = conv_vid_format.convert_to_h264(output_video_path)
        
    except Exception as e:
        print(f"error converting video format {e}")
    
    
    return output_video_path, csv_filepath , converted_output_video_path





if __name__ == "__main__":

    # output_video_path, csv_filepath , converted_output_video_path = process_video(video_file_path = video_file_path,class_name=class_name)

    pass