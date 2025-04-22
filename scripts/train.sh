#!/bin/bash/sh

python ./src/training/train_face_rec.py \
  --model_path ./models/yolo11s-cls.pt \
  --dataset_path ./data/training_dataset/yolo_dataset \
  --save_path ./models/yolov8_trained.pt
