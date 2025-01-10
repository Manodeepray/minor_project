import cv2
image_size = (300,300)


def create_detector():
    detector = cv2.FaceDetectorYN_create('models/face_detection_yunet_2023mar.onnx',
                            "", 
                            image_size,
                            score_threshold=0.9)
                            
    return detector       


def detect_faces(image):       
    
    h, w = image.shape[:2]

    # Set the input size for the detector
    detector.setInputSize((w, h))                
                    
    success, faces = detector.detect(image)

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
    detector = create_detector()
    image = cv2.imread("examples/input/iit.jpg")
    faces_coordinates = detect_faces(image)
    
