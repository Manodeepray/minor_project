
from ultralytics import YOLO
from . import face_det
import cv2
import time
import os



class FaceRecognizer:
    def __init__(self):
        self.model = YOLO('./models/yolov11l_trained.pt')  # Use the classification model (nano version)
        self.count = 0
        
        
    def load_model():
    
        return self.model
    
    def get_face_from_cropped(self,img_path):
        if img_path is not None:
            print("\n\n",img_path,"\n\n")
            results = self.model.predict(source = img_path , save = True , save_txt = True)
            for result in results:
                print(f" \n Top-1 Class: {result.names[result.probs.top1]}, Confidence: {result.probs.top1conf:.4f}")
            return result.names[result.probs.top1]
        else:
            return "None"
    
    def get_batch_inference(self,dir_path = ".data/processed/faces"):
        
        img_paths = os.listdir(dir_path)
        for i in range(len(img_paths)):
            img_path = os.path.join(dir_path,img_paths[i])
            img_paths[i] = img_path        
        print(img_paths)
        
        if img_paths == None or len(img_paths) == 0:
            frame_results = ["None"]
            return frame_results 
        
        else:
            results = self.model.predict(img_paths)
            
            names =  results[0].names
            # for result in results:
            #     names =  results.names
                # print(f"\n result : {names[result.probs.top1]} | conf : {result.probs.top1conf} ")
                
            frame_results = [names[result.probs.top1] for result in results]
            print(f"frame_results : {frame_results}")
            
            return frame_results



if __name__=="__main__":
    
    recognizer = FaceRecognizer()
    # detector = face_det.FaceDetector()

    # image = cv2.imread("examples/input/IMG-20241203-WA0008.jpg")
    # print("image : ",image)
    # img = "test"
    # cropped_face_folder = 'faces'
    # face_folder = detector.detect_faces(image ,cropped_face_folder  , clear_dir = True , img = img)



    # start_time = time.time()
    # for item in os.listdir(face_folder):
    #     img_path = os.path.join(face_folder,item)
    #     image = cv2.imread(path)
    #     image = cv2.resize(image, (224, 224))
    #     name = recognizer.get_face_from_cropped(img_path)    
    #     recognizer.count+=1
    # end_time = time.time()
    
    # print(f"\n Time Taken to recognize {recognizer.count} images : { end_time - start_time}s")
    
    frame_results = recognizer.get_batch_inference()
    print(frame_results)

    
