import argparse
from ultralytics import YOLO

def train_yolo_classification(
    model_path: str = "./models/yolo11s-cls.pt",
    dataset_path: str = "./data/training_dataset/yolo_dataset",
    save_path: str = "./models/yolov8_trained.pt",
    epochs: int = 1000,
    imgsz: int = 224,
    batch: int = 16,
    workers: int = 4,
    patience: int = 100000
):
    """
    Train a YOLO classification model.
    """
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    print(f"Starting training on dataset: {dataset_path}")
    model.train(
        data=dataset_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        patience=patience,
    )

    print(f"Saving trained model to: {save_path}")
    model.save(save_path)
    print("Training completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Train YOLO classification model.")
    parser.add_argument("--model_path", type=str, default="./models/yolo11s-cls.pt", help="Path to pre-trained model.")
    parser.add_argument("--dataset_path", type=str, default="./data/training_dataset/yolo_dataset", help="Path to training dataset.")
    parser.add_argument("--save_path", type=str, default="./models/yolov8_trained.pt", help="Path to save the trained model.")
    
    args = parser.parse_args()

    train_yolo_classification(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        save_path=args.save_path
    )

if __name__ == "__main__":
    main()
