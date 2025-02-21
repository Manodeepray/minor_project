from ultralytics import YOLO
# Initialize the YOLO model for classification
model = YOLO('/home/oreonmayo/minor_project/minor_project/models/yolo11s-cls.pt')  # Use the classification model (nano version)

# Train the model
model.train(
    data='/home/oreonmayo/minor_project/minor_project/yolo_dataset',  # Replace with the path to your dataset (e.g., './data')
    epochs=1000,             # Number of training epochs
    imgsz=224,               # Image size (YOLO recommends 224x224 for classification)
    batch=16,                # Batch size
    workers=4  ,             # Number of data loader workers
    patience = 100000
)

model.save('models/yolov8_trained.pt')


    
