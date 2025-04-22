import pandas
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

def analyze_attendance_csv(csv_filename):
    """
    Reads attendance data from a CSV file, counts the number of 'present' values for each student,
    and visualizes it in a bar plot.
    
    Parameters:
    - csv_filename (str): Path to the CSV file.
    
    Returns:
    - None
    """
    try:
        # Load the CSV file
        
        df = pd.read_csv(csv_filename)
        # Count the number of 'present' occurrences per person
        attendance_count = {}
        for column in df.columns:
            attendance_count[column] = (df[column] == "present").sum()

        # Convert to DataFrame for plotting
        attendance_df = pd.DataFrame(list(attendance_count.items()), columns=["Student", "frames_Present"])
        
        
        
        present_list = []
        attentiveness_list = []
        
        #attendance 
        for  i,row in enumerate(attendance_df['Student']):
               
            
            if attendance_df['frames_Present'][i] <= int(0.3*len(df)):
                print(f"Student {row} is absent |  observed in {attendance_df['frames_Present'][i]}/{len(df)} frames | criteria : {int(0.3*len(df))}/{len(df)} frames")
                present_list.append('absent')
            else:
                print(f"Student {row} has present |  observed in {attendance_df['frames_Present'][i]}/{len(df)} frames | criteria : {int(0.3*len(df))}/{len(df)} frames")
            
                present_list.append('present')
            
                
            # print(row,i)
                
        attendance_df['attendance'] = present_list
        
        #attentiveness
        for  i,row in enumerate(attendance_df['Student']):
               
                
            attntiv = (attendance_df['frames_Present'][i]/len(df))*100
            
            
            
            if attendance_df['attendance'][i] == 'present':
                
            
                attentiveness_list.append(attntiv)
            else:
                attentiveness_list.append("absent")

         
        
        
        attendance_df['attentiveness'] = attentiveness_list
        
        print(attendance_df)
        
        return attendance_df
    except Exception as e:
        print(f"Error: {e}")






# Example Usage

if __name__ == "__main__":
    csv_filename ="./data/database/database_attendance_csv/attendance.csv"
    attendance_df = analyze_attendance_csv(csv_filename)
