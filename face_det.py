import cv2
from config import config
import os
image_size = (config['IMGSIZE'] , config['IMGSIZE'])

class FaceDetector:
    def __init__(self):
        self.detector = cv2.FaceDetectorYN_create('models/face_detection_yunet_2023mar.onnx',
                                "", 
                                image_size,
                                score_threshold=0.9)
                                
        return None       


    # def detect_faces(self , image):       
        
    #     h, w = image.shape[:2]

    #     # Set the input size for the detector
    #     self.detector.setInputSize((w, h))                
                        
    #     success, faces = self.detector.detect(image)

    #     # Check if faces are detected
    #     if success and faces is not None:
    #         faces_coordinates = []
    #         for face in faces:
    #             # Unpack face data
    #             x, y, w, h, score = face[:5]
    #             cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    #             face_coords = list[int(x) , int(y) , int(x+w), int(y+h)]
    #             faces_coordinates.append(face_coords)
    #         output_path = "examples/output/output_with_faces.jpg"
    #         cv2.imwrite(output_path, image)
    #         print(f"Output image saved at: {output_path}")
            
    #     else:
    #         print("No faces detected.")
            
    #     print(faces_coordinates)
        
        
        
    #     return faces_coordinates
    
    
    

    
    def detect_faces(self, image):
        
        h, w = image.shape[:2]

        # Set the input size for the detector
        self.detector.setInputSize((w, h))

        # Detect faces
        success, faces = self.detector.detect(image)

        # Check if faces are detected
        if success and faces is not None:
            # Create the "faces" folder if it doesn't exist
            faces_coordinates = []
            faces_folder = "faces"
            clear_directory(faces_folder)
            os.makedirs(faces_folder, exist_ok=True)

            for i, face in enumerate(faces):
                # Extract face coordinates
                x, y, w, h, score = face[:5]

                # Ensure coordinates are within bounds
                x, y, w, h = int(x), int(y), int(w), int(h)
                if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
                    continue  # Skip if coordinates are invalid

                # Extract the face ROI (Region of Interest)
                face_roi = image[y:y + h, x:x + w]

                # Save the face as a separate image
                face_path = os.path.join(faces_folder, f"face_{i + 1}.jpg")
                cv2.imwrite(face_path, face_roi)
                print(f"Face {i + 1} saved at: {face_path}")
                cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                face_coords = list[int(x) , int(y) , int(x+w), int(y+h)]
                faces_coordinates.append(face_coords)
            output_path = "examples/output/output_with_faces.jpg"
            cv2.imwrite(output_path, image)
            print(f"Output image saved at: {output_path}")
            print(f"All detected faces are saved in the '{faces_folder}' folder.")
            return faces_folder
        else:
            print("No faces detected.")
            return None    

def clear_directory(directory_path):
    """
    Recursively deletes all contents of the given directory without deleting the directory itself.
    """
    if os.path.exists(directory_path):
        # Iterate over all files and subdirectories
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            # Remove files
            if os.path.isfile(item_path):
                os.remove(item_path)
                print(f"Deleted file: {item_path}")
            # Remove directories recursively
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Deleted folder and its contents: {item_path}")
    else:
        print(f"Directory '{directory_path}' does not exist.")

 

if __name__ == "__main__":
    detector = FaceDetector()
    image = cv2.imread("examples/input/IMG-20241203-WA0001.jpg")
    print("image : ",image)
    faces_coordinates = detector.detect_faces(image)
    
