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

import cv2
import os
import time

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Folder to store detected faces
faces_folder = "faces"
os.makedirs(faces_folder, exist_ok=True)

faces_recognized = []
start_time = time.time()

while True:
    ret, frame = cap.read()  # Capture a frame from the webcam

    if not ret:
        print("Failed to grab frame.")
        break

    # Display the live feed
    cv2.imshow("Webcam Feed", frame)

    # Detect faces and save them in the folder
    detected_faces_folder = detector.detect_faces(frame)

    # Recognize each face in the folder
    for face_image in os.listdir(detected_faces_folder):
        img_path = os.path.join(detected_faces_folder, face_image)

        # Process and extract embeddings
        processed_image = embeddings.process_image(path=img_path)
        test_embedding = embeddings.get_embeddings(model, processed_image)

        # Perform face recognition
        similarities, pred_face = embeddings.recognize_face(test_embedding, database_embeddings)
        print(f"For image: {img_path}")
        print(f"Similarities: {similarities}")
        print(f"Predicted face: {pred_face}")
        faces_recognized.append(pred_face)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

# Print recognized faces
print(faces_recognized)

end_time = time.time()
duration = end_time - start_time
print(f"Function duration: {duration:.4f} seconds")
