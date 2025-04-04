import asyncio
import asyncpg
import pandas as pd
from enum import Enum

# PostgreSQL connection details
DB_URL = "postgresql://minor_owner:npg_3xNhiAkr9OUE@ep-silent-feather-a10qi6gs-pooler.ap-southeast-1.aws.neon.tech/minor?sslmode=require"

# CSV file path
CSV_FILE = "/teamspace/studios/this_studio/minor_project/database_attendance_csv/minor_demo.csv"

# Define ENUM for Attendance Status
class Status(Enum):
    PRESENT = "Present"
    ABSENT = "Absent"
    LATE = "Late"

# Mapping of status variations to ENUM values
STATUS_MAPPING = {
    "present": Status.PRESENT.value,
    "status.present": Status.PRESENT.value,  # Handling status.present key issue
    "absent": Status.ABSENT.value,
    "status.absent": Status.ABSENT.value,  # Handling status.absent key issue
    "late": Status.LATE.value,
}

async def upload_csv():
    # Load CSV into a DataFrame
    df = pd.read_csv(CSV_FILE)

    # Ensure required columns exist
    required_columns = {"Student", "attentiveness", "attendance"}
    if not required_columns.issubset(df.columns):
        print(f"❌ CSV is missing required columns: {required_columns - set(df.columns)}")
        return

    # Connect to PostgreSQL
    conn = await asyncpg.connect(DB_URL)

    async with conn.transaction():
        for _, row in df.iterrows():
            # Process attentiveness safely
            try:
                row["attentiveness"] = round(float(row["attentiveness"])) if pd.notna(row["attentiveness"]) else None
            except ValueError:
                row["attentiveness"] = None  # Handle conversion errors

            # Convert attendance to ENUM value with default to "Absent"
            row["attendance"] = STATUS_MAPPING.get(row["attendance"].strip().lower(), Status.ABSENT.value)

            # Insert into PostgreSQL with conflict handling
            await conn.execute(
                """
                INSERT INTO "AttendanceRecord" ("studentID", "attentiveness", "status")
                VALUES ($1, $2, $3)
                ON CONFLICT ("studentID") DO UPDATE 
                SET "attentiveness" = EXCLUDED."attentiveness", 
                    "status" = EXCLUDED."status";
                """,
                row["Student"],
                row["attentiveness"],
                row["attendance"]
            )

    await conn.close()
    print("✅ CSV uploaded successfully!")

# Run async function
asyncio.run(upload_csv())
