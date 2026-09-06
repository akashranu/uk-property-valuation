import time
import pandas as pd
from sqlalchemy import create_engine

# Database Connection Configuration
DB_USER = "akash_property_ml"
DB_PASS = "Oligopolistic2"  # Replace with your actual user password
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "uk_property_db"

DATABASE_URI = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
CSV_FILE_PATH = "data/pp-2024.csv"

# Define Explicit Schema Columns (matching the Land Registry technical spec)
COLUMN_NAMES = [
    "transaction_id",
    "price",
    "transfer_date",
    "postcode",
    "property_type",
    "old_new",
    "duration",
    "paon",
    "saon",
    "street",
    "locality",
    "town_city",
    "district",
    "county",
    "ppd_category_type",
    "record_status",
]


def ingest_data(chunk_size: int = 50000):
    print("Connecting to MySQL...")
    engine = create_engine(DATABASE_URI)

    print(f"Starting ingestion from {CSV_FILE_PATH} in chunks of {chunk_size} rows...")
    start_time = time.time()
    total_rows = 0

    try:
        # Stream CSV in chunks to keep memory usage low
        chunk_iter = pd.read_csv(
            CSV_FILE_PATH,
            names=COLUMN_NAMES,
            header=None,
            chunksize=chunk_size,
            dtype={
                "transaction_id": "string",
                "property_type": "string",
                "old_new": "string",
                "duration": "string",
                "ppd_category_type": "string",
                "record_status": "string",
                "postcode": "string",
            },
            parse_dates=["transfer_date"],
        )

        for i, chunk in enumerate(chunk_iter, 1):
            # Clean transfer_date format to standard YYYY-MM-DD
            chunk["transfer_date"] = chunk["transfer_date"].dt.strftime("%Y-%m-%d")

            # Append chunk into the existing MySQL table
            chunk.to_sql(
                name="raw_property_sales",
                con=engine,
                if_exists="append",
                index=False,
                method="multi",  # Batch insert for high speed
            )

            total_rows += len(chunk)
            print(f"Batch {i} written: {total_rows:,} total rows inserted...")

        elapsed = time.time() - start_time
        print(f"Ingestion complete: {total_rows:,} rows loaded in {elapsed:.2f} seconds.")

    except Exception as e:
        print(f"Ingestion failed: {e}")


if __name__ == "__main__":
    ingest_data(chunk_size=50000)