import face_det
import cv2
import time
import os
import face_recog


########################################################process single images###################################################################



def process_image(image_path , detector , recognizer ,output_frames_dir = "frames_output"):

    start_time = time.time()
        
        
    

    image = cv2.imread(image_path)
    print("image : ",image)

    cropped_face_folder = 'faces'
    img = "test_main"
    face_folder , _ , _= detector.detect_faces(image ,cropped_face_folder  , clear_dir = True , img = img,
                                               output_frames_dir=output_frames_dir)

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



################################################### WEBCAM #########################################$$$$$$$$


import cv2
import os
import time

def process_video_from_webcam(detector, recognizer, output_frames_dir="frames_output"):
    """
    Detects faces from webcam frames and classifies the face in real time.
    """
    attendance = []
    # Create folder to save detected faces
    os.makedirs(output_frames_dir, exist_ok=True)

    # Open the webcam (0 for default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30  # Set default FPS if unknown
    frame_interval = int(fps * 0.5)  # Process frames every 0.5 seconds
    frame_count = 0
    

    print("Press 'q' to stop the webcam feed.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from webcam.")
            break

        # Process every nth frame
        if frame_count % frame_interval == 0:
            cropped_face_folder = "faces"
            print("Processing frame...")

            # Detect faces
            face_folder, _, _ = detector.detect_faces(frame, cropped_face_folder,
                                                      clear_dir=True,
                                                      img=frame_count,
                                                      output_frames_dir=output_frames_dir)

            if face_folder.lower() == "none":
                print("No faces detected.")
                continue

            faces_recognized = []
            if len(os.listdir(face_folder)) != 0:
                for image in os.listdir(face_folder):
                    img_path = os.path.join(face_folder, image)
                    name = recognizer.get_face_from_cropped(img_path)
                    recognizer.count += 1
                    faces_recognized.append(name)

                print(f"Recognized faces: {faces_recognized}")

            attendance.append(faces_recognized)

        frame_count += 1
        print(f"Frame {frame_count}: {faces_recognized}")

        # Display the webcam feed
        cv2.imshow("Webcam Feed", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    # Print attendance results
    print("\nFinal Attendance List:", attendance)





################################################### video ###################################################

def process_video(video_path,detector , recognizer , output_frames_dir="frames_output" ):
    
    """
    detects faces from video frames and classifies the face
    """
    # Create folder to save faces
    os.makedirs(output_frames_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    frame_interval = int(fps * 0.5)  # Interval for processing frames (every 0.5 seconds)
    frame_count = 0
    faces_recognized = []
    start_time = time.time()
    attendance = []
    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video reached or unable to read frame.")
            break

        # Process frames at the specified interval
        if frame_count % frame_interval == 0:
            # Detect faces in the current frame

            cropped_face_folder = 'faces'
            img = "test_video_main"
            print(frame)
            face_folder ,_ ,_= detector.detect_faces(frame ,cropped_face_folder  , clear_dir = True , img = frame_count, output_frames_dir= output_frames_dir)
            
            
            
            
            if face_folder.lower() == 'none':
                cap.release()
                break
            print("\n\n",len(os.listdir(face_folder)),os.listdir(face_folder))
            faces_recognized = []
            if len(os.listdir(face_folder) ) != 0:
                for image in os.listdir(face_folder):
                    img_path = os.path.join(face_folder , image)
                    
                    name = recognizer.get_face_from_cropped(img_path)    
                    recognizer.count+=1
                    
                    
                    
                    faces_recognized.append(name)

                print(faces_recognized)
        attendance.append(faces_recognized)        
        # Increment the frame count
        frame_count += 1
        print(f"For frame no. : {frame_count} | faces recognized : {faces_recognized} \n\n")
    # Clean up
    cap.release()

    # Print results
    

    end_time = time.time()
    duration = end_time - start_time
    print("faces_recognized :",faces_recognized)
    print("No. of frames : ",frame_count)
    print(f"Function duration: {duration:.4f} seconds")
    print("Attendance : ",attendance)


if __name__ == "__main__":
        

    detector = face_det.FaceDetector()
    recognizer = face_recog.FaceRecognizer()
    
    # # video_path = "/home/oreonmayo/minor_project/minor_project/examples/input/29_01_25.mp4"
    video_path = "examples/input_examples/WIN_20250119_00_46_21_Pro.mp4"
    process_video(video_path, detector , recognizer )

    # image_path = "/home/oreonmayo/minor_project/minor_project/examples/input/IMG-20241203-WA0008.jpg" 
    # process_image(image_path , detector , recognizer )
