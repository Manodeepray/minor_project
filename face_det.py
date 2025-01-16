import cv2
from config import config

image_size = (config['IMGSIZE'] , config['IMGSIZE'])

class FaceDetector:
    def __init__(self):
        self.detector = cv2.FaceDetectorYN_create('models/face_detection_yunet_2023mar.onnx',
                                "", 
                                image_size,
                                score_threshold=0.9)
                                
        return None       


    def detect_faces(self , image):       
        
        h, w = image.shape[:2]

        # Set the input size for the detector
        self.detector.setInputSize((w, h))                
                        
        success, faces = self.detector.detect(image)

        # Check if faces are detected
        if success and faces is not None:
            faces_coordinates = []
            for face in faces:
                # Unpack face data
                x, y, w, h, score = face[:5]
                cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                face_coords = list[int(x) , int(y) , int(x+w), int(y+h)]
                faces_coordinates.append(face_coords)
            output_path = "examples/output/output_with_faces.jpg"
            cv2.imwrite(output_path, image)
            print(f"Output image saved at: {output_path}")
            
        else:
            print("No faces detected.")
            
        print(faces_coordinates)
        
        return faces_coordinates


if __name__ == "__main__":
    detector = FaceDetector()
    image = cv2.imread("examples/input/IMG-20241203-WA0001.jpg")
    print("image : ",image)
    faces_coordinates = detector.detect_faces(image)
    
