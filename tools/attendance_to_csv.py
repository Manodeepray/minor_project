import csv

def save_dicts_to_csv(dict_list, csv_filename="output.csv"):
    """
    Saves a list of dictionaries to a CSV file.
    
    Parameters:
    - dict_list (list): List of dictionaries.
    - csv_filename (str): Output CSV file name (default: "output.csv").
    
    Returns:
    - None
    """
    if not dict_list:
        print("Error: The list is empty. No CSV file created.")
        return
    
    # Extract all unique keys to form the header row
    all_keys = set()
    for entry in dict_list:
        all_keys.update(entry.keys())

    # Write to CSV
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=all_keys)
        writer.writeheader()  # Write the header row
        
        for entry in dict_list:
            writer.writerow(entry)  # Write each dictionary as a row

    print(f"CSV file '{csv_filename}' created successfully!")
    
    return csv_filename

if __name__=="__main__":
    # Example Usage
    attendance_list = [
        {'manodeep': 'present', 'akash': None, 'harshit': 'present', 'hasan': 'present', 
        'ashmit': None, 'devanshu': None, 'mani': None, 'arunoday': None, 'himanshu': 'present'},
        {'manodeep': 'present', 'akash': None, 'harshit': 'present', 'hasan': 'present', 
        'ashmit': None, 'devanshu': None, 'mani': None, 'arunoday': None, 'himanshu': 'present'},
        {'manodeep': 'present', 'akash': None, 'harshit': 'present', 'hasan': 'present', 
        'ashmit': None, 'devanshu': None, 'mani': None, 'arunoday': None, 'himanshu': 'present'}
    ]

    save_dicts_to_csv(attendance_list, "./attendance_csv/attendance.csv")
