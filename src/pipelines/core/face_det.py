import cv2
from .config.config import config
import os

import shutil

image_size = (config['IMGSIZE'] , config['IMGSIZE'])

class FaceDetector:
    def __init__(self):
        self.detector = cv2.FaceDetectorYN_create('./models/face_detection_yunet_2023mar.onnx',
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
    
    
    

    
    def detect_faces(self, image , output_dir , clear_dir ,img ,output_frames_dir):
        """
        detects faces from image and saves it in output folder
        
        Args:
        image = input image data from cv.imread or pil
        output_dir =  folder to save the cropped faces
        clear_dir = clears the folder to prevent residual images
        img = name of the person
        
        """
        
        h, w = image.shape[:2]
        _frame = image

        # Set the input size for the detector
        self.detector.setInputSize((w, h))
        
        try:
            # Detect faces
            success, faces = self.detector.detect(image)
            print(f" Faces detected - sucess : {success} | faces : {faces}\n")
        except:

            print("faces not detected\n")
            # break
        
        # success, faces = self.detector.detect(image)
        
        
        # Check if faces are detected
        if success and faces is not None:
            print("success and faces not None \n")
        
        
        
            # Create the "faces" folder if it doesn't exist
            faces_coordinates = []
            faces_roi = []
            
            faces_folder = output_dir
        
        
            if clear_dir == True:
                clear_directory(faces_folder)
                
            os.makedirs(faces_folder, exist_ok=True)
        
        
        
            for i, face in enumerate(faces):
                # Extract face coordinates
                x, y, w, h, score = face[:5]
                print(f"x, y, w, h :{x, y, w, h} | score : {score}")
                # Ensure coordinates are within bounds
                x, y, w, h = int(x), int(y), int(w), int(h)
        
        
        
                if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
                    continue  # Skip if coordinates are invalid


                # Extract the face ROI (Region of Interest)
                face_roi = image[y+2:y + h+2, x-2:x + w+2]

                # Save the face as a separate image
                face_path = os.path.join(faces_folder, f"{img}_face_{i + 1}.jpg")

                # face_roi = cv2.resize(face_roi, dsize = (224,224))

                cv2.imwrite(face_path, face_roi)


                print(f"Face {i + 1} saved at: {face_path}")


                cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                
                face_coords = list[int(x) , int(y) , int(x+w), int(y+h)]
                                
                faces_coordinates.append(face_coords)

                faces_roi.append(face_roi)
                
                
                
            output_path = f"{output_frames_dir}/faceDet_output_{img}.jpg"
            
            cv2.imwrite(output_path, image)
            
            print(f"Output image saved at: {output_path}")
            
            print(f"All detected faces : {len(faces)}  saved in the '{faces_folder}' folder.")
            
            return faces_folder , faces_coordinates , faces
        
        else:
            output_path = f"{output_frames_dir}/faceDet_output_{img}.jpg"
            
            cv2.imwrite(output_path, _frame)
         
            
            print(f"Output image saved at: {output_path}")
            
            print("No faces detected.\n")
            
            return "None" , [] ,[]






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





def get_train_dataset():
    """
     Iterates through the dataset of images and saves the detected face for model training
     
     
    """
    detector = FaceDetector()
    
    dataset_dir = "./dataset"

    new_dataset_dir = "./data/training_dataset/yolo_dataset/train"
    
    
    os.makedirs(new_dataset_dir , exist_ok = True)
    
    
    for item in os.listdir(dataset_dir):
        
        dir_path = os.path.join( dataset_dir,item )
        print("dir_path : ",dir_path)
        
        new_dir_path = os.path.join(new_dataset_dir , item)
        print("new_dir_path : ",new_dir_path)
        
        os.makedirs( new_dir_path, exist_ok= True)
        
        
        for img in os.listdir(dir_path):
        
            img_path = os.path.join( dir_path,img)
        
            print("img_path :",img_path)
        
            image = cv2.imread(img_path)
        
            face = detector.detect_faces(image , new_dir_path , clear_dir = False , img = img)
    
if __name__ == "__main__":
    pass
    # testing the detector
    # detector = FaceDetector()

    # image = cv2.imread("examples/input/IMG-20241203-WA0008.jpg")
    # print("image : ",image)
    # img = "1"
    
    # _,_,_ = detector.detect_faces(image , 'faces' , clear_dir = True , img = img)
    
    # get_train_dataset()        
    
