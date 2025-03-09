import os
import shutil



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