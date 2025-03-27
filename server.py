from flask import Flask, request
import os
import time
from tools import separation
app = Flask(__name__)

# Define the folder where videos will be saved
UPLOAD_FOLDER = 'uploaded_videos'
def clean_dir(dir):
    files = os.listdir(dir)
    for file in files:
        filepath = os.path.join(dir , file)
        print(f"deleting {filepath}")
        os.remove(filepath)
        

    return None
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define the route to handle the file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    clean_dir(UPLOAD_FOLDER)

    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file', 400
    
    # Generate a unique filename based on timestamp and original filename
    timestamp = int(time.time())  # Generate a unique timestamp for the file name
    filename = f"uploaded_{file.filename}"
    
    
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    # Save the uploaded file to the server
    file.save(file_path)
    
    
    print(f'File uploaded successfully: {filename}')
    separation.separate_video_audio(file_path)

    return f'File uploaded successfully: {filename}', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
