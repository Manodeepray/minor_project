from deepface import DeepFace





if __name__ == "__main__":
    # verification = DeepFace.verify(img1_path = "yolo_dataset/train/manodeep/face_1.jpg",
    #                                img2_path = "yolo_dataset/train/manodeep/Screenshot 2025-01-01 195947.png_face_1.jpg",
    #                                model_name = "OpenFace")
    # print(f"Verification : {verification}")
    
    dfs = DeepFace.find(
    img_path = "faces/45_face_1.jpg",
    db_path = "yolo_dataset/train", 
    model_name = "OpenFace",
    )
    print(dfs)