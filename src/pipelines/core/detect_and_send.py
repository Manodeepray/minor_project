import face_det
import cv2
import requests
import numpy as np
import time
import os

CLOUD_URL = "http://172.30.185.24:8000"













def encode_images(faces):
    encoded_faces = []
    files = []
    for face in faces:
        _, img_encoded = cv2.imencode('.jpg', face)
        print(f"img_encoded : {img_encoded}")
        encoded_faces.append(img_encoded)

    for  i , img_encoded in enumerate(encoded_faces):
        file= {f'file_{i}': (f"face_{i}.jpg", img_encoded.tobytes(), "image/jpeg")}
        files.append(file)
    print(f"Image encoded | length : {len(files)} \n")
    return files


def send_batch(files):
    # files = [("files", (f, open(os.path.join(face_folder, f), "rb"), "image/jpeg")) for f in os.listdir(face_folder)]
    
    if files:
        print(f"files detected | trying to upload batch | \n\n files : {files} \n\n")
        try:
            response = requests.post(CLOUD_URL, files=files)
            print(response.json())
            print(f"batch response sent: {response} to {CLOUD_URL}")
        except Exception as e:  
            print(f"error uploading the detected faces in frame as batch image files to {CLOUD_URL}")        
    





def process_image_and_semd(image_path , detector):

    start_time = time.time()
    
    image = cv2.imread(image_path)
    
    print("image : ",image)

    cropped_face_folder = './data/processed/faces'
    
    img = "test_main"
    
    faces_folder , faces_coordinates , faces = detector.detect_faces(image ,cropped_face_folder  , clear_dir = True , img = img)

    if faces_folder.lower() == 'none':
        cap.release()
        # break
    
    
    files = encode_images(faces)
    send_batch(files)


    end_time = time.time()

    duration = end_time - start_time

    print(f"Function duration: {duration:.4f} seconds")








def process_video_and_send(video_path,detector  , output_faces_folder="./data/processed/video_output" ):
    
    """
    detects faces , resizes and then sends to online server
    
    """
    
    
    # Create folder to save faces
    os.makedirs(output_faces_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    frame_interval = int(fps * 0.5)  # Interval for processing frames (every 0.5 seconds)
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video reached or unable to read frame.")
            break

        # Process frames at the specified interval
        if frame_count % frame_interval == 0:
            # Detect faces in the current frame

            cropped_face_folder = './data/processed/faces'
            img = "test_video_main"
            faces_folder , faces_coordinates , faces = detector.detect_faces(frame ,cropped_face_folder  , clear_dir = True , img = frame_count)
            
            
            if faces_folder.lower() == 'none':
                cap.release()
                break
            
            
            files = encode_images(faces)
            send_batch(files)
           
            

    cap.release()

    # Print results
    

    end_time = time.time()
    duration = end_time - start_time
    print("No. of frames : ",frame_count)
    print(f"Function duration: {duration:.4f} seconds")


if __name__=="__main__":
    
    

    detector = face_det.FaceDetector()
    # video_path = "examples/input/WIN_20250119_00_46_21_Pro.mp4"
    # process_video_and_send(video_path, detector )


    image_path = "./examples/input/IMG-20241203-WA0008.jpg" 
    process_image_and_semd(image_path , detector )