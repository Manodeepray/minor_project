import face_det
import cv2
import time
import os
import face_recog




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




def process_batch_faces(image_path , detector , recognizer):

    start_time = time.time()
        
        
    

    image = cv2.imread(image_path)
    # print("image : ",image)

    cropped_face_folder = './faces'
    img = "test_main"
    face_folder , _ , _= detector.detect_faces(image ,cropped_face_folder  , clear_dir = True , img = img)

    faces_recognized = recognizer.get_batch_inference(dir_path = cropped_face_folder)

    # print(faces_recognized)

    end_time = time.time()

    duration = end_time - start_time

    # print(f"Function duration: {duration:.4f} seconds")
    
    return faces_recognized



def get_frame_attendance(image_path,detector , recognizer):
    frame_attendance_sheet = {'manodeep':None,
                        'akash':None,
                        'harshit':None,
                        'hasan':None,
                        'ashmit':None,
                        'devanshu':None,
                        'mani':None,
                        'arunoday':None,
                        'himanshu':None,
                        }
    faces_recognized = process_batch_faces(image_path = "/home/oreonmayo/minor_project/minor_project/examples/input/IMG-20241203-WA0003.jpg",
               detector = detector,
               recognizer = recognizer )
    
    
    for name in faces_recognized:
        if name in frame_attendance_sheet:
            frame_attendance_sheet[name] =  "present"
    
            
    print("\n frame_attendance_sheet : ",frame_attendance_sheet)
    
    
    return frame_attendance_sheet



if __name__== "__main__":
    
    
    
    
    detector = face_det.FaceDetector()
    recognizer = face_recog.FaceRecognizer()
    
    
    
    # faces_recognized = process_image(image_path = "/home/oreonmayo/minor_project/minor_project/examples/input/IMG-20241203-WA0003.jpg",
    #            detector = detector,
    #            recognizer = recognizer )
    
    frame_attendance_sheet = get_frame_attendance(image_path = "/home/oreonmayo/minor_project/minor_project/examples/input/IMG-20241203-WA0003.jpg",
               detector = detector,
               recognizer = recognizer)
   
    
    
    
    
    
    