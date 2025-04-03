from flask import Flask, request
import os
import time
from tools import separation  # Ensure this module exists

app = Flask(__name__)

UPLOAD_FOLDER = 'uploaded_videos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def clean_dir(directory: str) -> None:
    """Deletes all files in the specified directory."""
    for file in os.listdir(directory):
        filepath = os.path.join(directory, file)
        print(f"Deleting {filepath}")
        os.remove(filepath)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file uploads."""
    clean_dir(UPLOAD_FOLDER)

    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Generate a unique filename
    filename = f"uploaded_{int(time.time())}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    # Save the uploaded file
    file.save(file_path)
    print(f'File uploaded successfully: {filename}')

    # Process video
    separation.separate_video_audio(video_path=file_path)

    return f'File uploaded successfully: {filename}', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
