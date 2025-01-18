import face_det
import cv2
import embeddings
import time
import os

database_directory_path = "dataset"
model = embeddings.get_model()
database_embeddings = embeddings.get_database(model,database_directory_path)

detector = face_det.FaceDetector()


# start_time = time.time()
    
    
# test_class_image_path = "examples/input/IMG-20241203-WA0004.jpg" 

# image = cv2.imread(test_class_image_path)
# print("image : ",image)

# faces_folder = detector.detect_faces(image)

# faces_recognized = []

# for image in os.listdir(faces_folder):
#     img_path = os.path.join(faces_folder , image)
    
#     image = embeddings.process_image(path = img_path)
#     test_embedding = embeddings.get_embeddings(model , image)

#     similarities , pred_face = embeddings.recognize_face(test_embedding, database_embeddings)  
#     print("for image :",img_path)
#     print("similarities",similarities)
#     print("predicted face :",pred_face)
#     faces_recognized.append(pred_face)

# print(faces_recognized)

# end_time = time.time()

# duration = end_time - start_time

# print(f"Function duration: {duration:.4f} seconds")



################################################### WEBCAM #########################################








def process_video(video_path,output_faces_folder="faces"):
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
            faces_folder = detector.detect_faces(frame)
            if faces_folder.lower() == 'none':
                cap.release()
                break
            print("\n\n",len(os.listdir(faces_folder)),os.listdir(faces_folder))
            faces_recognized = []
            if len(os.listdir(faces_folder) ) != 0:
                for image in os.listdir(faces_folder):
                    img_path = os.path.join(faces_folder , image)
                    
                    image = embeddings.process_image(path = img_path)
                    test_embedding = embeddings.get_embeddings(model , image)

                    similarities , pred_face = embeddings.recognize_face(test_embedding, database_embeddings)  
                    print("for image :",img_path)
                    print("similarities",similarities)
                    print("predicted face :",pred_face)
                    faces_recognized.append(pred_face)

                print(faces_recognized)
        attendance.append(faces_recognized)        
        # Increment the frame count
        frame_count += 1

    # Clean up
    cap.release()

    # Print results
    

    end_time = time.time()
    duration = end_time - start_time
    print("faces_recognized :",faces_recognized)
    print("No. of frames : ",frame_count)
    print(f"Function duration: {duration:.4f} seconds")
    print("Attendance : ",attendance)

video_path = "examples/input/WIN_20250119_00_46_21_Pro.mp4"
process_video(video_path)