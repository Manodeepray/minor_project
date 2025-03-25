import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import os

# Load .env file


def setup():
    load_dotenv()

    MONGO_DB_URI = os.getenv("MONGO_DB_URI")
    MONGO_DB_PASSWORD = os.getenv("MONGO_DB_PASSWORD")

    # MongoDB connection details
    mongo_uri = MONGO_DB_URI.replace("<db_password>" ,MONGO_DB_PASSWORD )  # Replace with your MongoDB URI
    print(f"\n mongo uri : {mongo_uri} \n")
    database_name = "minor_project"
    collection_name = "attendance_csv"

    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client[database_name]
    collection = db[collection_name]

    return collection

# Folder containing CSV files
# csv_file_path = "C:/Users/KIIT/projects/minor_project_/minor_project-main/attendance_csv/attendance.csv"

def send_csv_Mongodb(csv_file_path):
    
    print("setting up")
    
    collection = setup()
    
    
    print(f"Processing {csv_file_path}...")


        
    file_name = os.path.basename(csv_file_path)

    # Remove the file extension (e.g., "class1")
    class_period = os.path.splitext(file_name)[0]

    print(f"Extracted class name: {class_period}")


        # Read CSV into a DataFrame
    df = pd.read_csv(csv_file_path)
    df["class_period"] = class_period

    # Convert DataFrame to a list of dictionaries (one dict per row)
    data = df.to_dict("records")

    # Insert data into MongoDB
    collection.insert_many(data)
    print(f"Inserted {len(data)} records from {csv_file_path}")



    print("CSV files processed and data inserted into MongoDB.")

    
# send_csv(csv_file_path=csv_file_path)    