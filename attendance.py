import face_det
import cv2
import time
import os
import face_recog

from tools.attendance_to_csv import save_dicts_to_csv
from tools.convert_to_video import frames_to_video


def process_image(image_path , detector , recognizer):

    start_time = time.time()
        
        
    

    image = cv2.imread(image_path)
    print("image : ",image)

    cropped_face_folder = 'faces'
    img = "test_main"
    face_folder , _ , _= detector.detect_faces(image ,cropped_face_folder  , clear_dir = True , img = img)

    faces_recognized = []

    for item in os.listdir(face_folder):
        
        img_path = os.path.join(face_folder,item)
        # image = cv2.imread(path)
        # image = cv2.resize(image, (224, 224))
        name = recognizer.get_face_from_cropped(img_path)    
        recognizer.count+=1
        
        faces_recognized.append(name)

    print(faces_recognized)

    end_time = time.time()

    duration = end_time - start_time

    print(f"Function duration: {duration:.4f} seconds")
    
    return faces_recognized




def process_batch_faces(image_path , detector , recognizer,output_frames_dir,frame_count):

    start_time = time.time()
        
        
    

    image = cv2.imread(image_path)
    # print("image : ",image)

    cropped_face_folder = './faces'
    img = frame_count
    face_folder , frame_faces_coordinates , _= detector.detect_faces(image ,cropped_face_folder  , clear_dir = True , img = img,
                                               output_frames_dir=output_frames_dir)

    faces_recognized = recognizer.get_batch_inference(dir_path = cropped_face_folder)

    # print(faces_recognized)

    end_time = time.time()

    duration = end_time - start_time

    # print(f"Function duration: {duration:.4f} seconds")
    
    return faces_recognized ,frame_faces_coordinates 



def get_frame_attendance(image_path,detector , recognizer,frame_count,output_frames_dir="frames_output" ):
    frame_attendance_sheet = {'manodeep':"absent",
                        'akash':"absent",
                        'harshit':"absent",
                        'hasan':"absent",
                        'ashmit':"absent",
                        'devanshu':"absent",
                        'mani':"absent",
                        'arunoday':"absent",
                        'himanshu':"absent",
                        }
    faces_recognized ,frame_faces_coordinates  = process_batch_faces(image_path = image_path,
               detector = detector,
               recognizer = recognizer,
               output_frames_dir=output_frames_dir,
               frame_count=frame_count)
    
    
    for name in faces_recognized:
        if name in frame_attendance_sheet:
            frame_attendance_sheet[name] =  "present"
    
            
    print("\n frame_attendance_sheet : ",frame_attendance_sheet)
    
    
    return frame_attendance_sheet , frame_faces_coordinates





################################################## attendance from video#######################################

import cv2
import os
import time
from tools.clear_dir import clear_directory


def get_attendance_from_video(video_path, output_frames_dir="frames_output"):
    
    #clear the recorded frames from previous run
    clear_directory(output_frames_dir)
    
    
    
    
    
    # Initialize detector and recognizer models
    detector = face_det.FaceDetector()
    recognizer = face_recog.FaceRecognizer()
    
    
    #create buffer
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)  # Create folder if it doesn't exist
    os.makedirs(output_frames_dir, exist_ok=True)  # Ensure output directory exists
    
    
    
    processing_frame_count = 0
    Attendance = []
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return Attendance, output_frames_dir

    fps = cap.get(cv2.CAP_PROP_FPS)  # Get original FPS
    frame_skip = int(fps * 1)  # Skip frames to achieve 0.5 FPS (1 frame every 2 sec)

    frame_idx = 0  # Track the current frame index
    
    while True:
        ret, frame = cap.read()
        
        if not ret:  # Break loop if video ends
            break

        if frame_idx % frame_skip == 0:  # Process every nth frame
            
            
            frame_filename = os.path.join(temp_dir, f"frame_{processing_frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            
            
            
            frame_attendance_sheet, _ = get_frame_attendance(
                image_path = frame_filename,
                detector = detector,recognizer= recognizer,
                output_frames_dir = output_frames_dir,
                frame_count= frame_idx
            )  # Process frame
            
            Attendance.append(frame_attendance_sheet)
            
            processing_frame_count += 1  # Increment processed frame count
            
            os.remove(frame_filename)  # Clean up temp file
        
        frame_idx += 1  # Increment frame index
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    return Attendance, output_frames_dir


if __name__== "__main__":
    
    
    
    
    # detector = face_det.FaceDetector()
    # recognizer = face_recog.FaceRecognizer()
    
    
    
    # faces_recognized = process_image(image_path = "/home/oreonmayo/minor_project/minor_project/examples/input/IMG-20241203-WA0003.jpg",
    #            detector = detector,
    #            recognizer = recognizer )
    
    # frame_attendance_sheet = get_frame_attendance(image_path = "/home/oreonmayo/minor_project/minor_project/examples/input/IMG-20241203-WA0003.jpg",
    #            detector = detector,
    #            recognizer = recognizer)
    
    
    Attendance , output_frames_dir = get_attendance_from_video(video_path = "/home/oreonmayo/minor_project/minor_project/recorded_videos/29_01_25.mp4")
    save_dicts_to_csv(dict_list = Attendance ,csv_filename = "./attendance_csv/attendance.csv" )
    frames_to_video(output_frames_dir,"./processed_video/output_video.mp4", fps=1)
    
    
    
    
    
    