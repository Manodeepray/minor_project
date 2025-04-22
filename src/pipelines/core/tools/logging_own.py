import os

LOG_FILE = "./logs/processed_videos.txt"

def load_processed_videos():
    print("""Load processed video filenames from the log file.""")
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, "r") as file:
        return set(line.strip() for line in file)

def save_processed_video(video_name):
    print(f"""Append processed video filename to the log file.{video_name}""")
    with open(LOG_FILE, "a") as file:
        file.write(video_name + "\n")


if __name__ == "__main__":
    pass