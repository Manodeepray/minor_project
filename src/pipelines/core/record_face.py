import cv2
import os

# Define the folder where videos will be saved
def record_class(output_dir = "./data/raw/recorded_videos"):

    save_folder = output_dir
    os.makedirs(save_folder, exist_ok=True)  # Create folder if it doesn't exist

    # Define the video filename with a unique timestamp
    video_file_path = os.path.join(save_folder, "output.mp4")  # Change to .mp4 if needed

    # Open webcam
    cap = cv2.VideoCapture(0)  # 0 for default webcam

    # Get video properties
    frame_width = int(cap.get(3))  # Width
    frame_height = int(cap.get(4))  # Height
    fps = 20  # Frames per second (adjust as needed)

    # Define video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
    out = cv2.VideoWriter(video_file_path, fourcc, fps, (frame_width, frame_height))

    print("[INFO] Recording... Press 'q' to stop.")

    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            break

        out.write(frame)  # Save frame to file
        cv2.imshow("Recording...", frame)  # Display the recording window

        # Press 'q' to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Video saved as: {video_file_path}")
    return video_file_path   


if __name__ =="__main__":
    
    video_file_path = record_class(output_dir = "./data/raw/recorded_videos")