import cv2
import os
from natsort import natsorted  # Ensures proper ordering of numbered images

def frames_to_video(image_dir, output_video_path, fps=30):
    # Get all .jpg files in the directory
    images = [img for img in os.listdir(image_dir) if img.endswith(".jpg")]
    
    if not images:
        print("No .jpg images found in the directory!")
        return
    
    images = natsorted(images)  # Sort filenames naturally (e.g., frame_1.jpg, frame_2.jpg,...)
    
    first_image_path = os.path.join(image_dir, images[0])
    first_frame = cv2.imread(first_image_path)
    
    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        frame_path = os.path.join(image_dir, image)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Skipping corrupted/missing file: {image}")
            continue
        
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved as: {output_video_path}")

if __name__=="__main__":
# Example usage
    frames_to_video("./frames_output", "./processed_video/output_video.mp4", fps=1)
