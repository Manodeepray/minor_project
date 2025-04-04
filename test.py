from sqlalchemy import create_engine, text

# Define the database URL directly
DATABASE_URL = "postgresql://user:password@localhost:5432/mydatabase"

# Create the database engine
engine = create_engine(DATABASE_URL)

# Test the connection
with engine.connect() as connection:
    result = connection.execute(text("SELECT * FROM AttendanceRecord"))
    for row in result:
        print(row)  # Print each attendance record
