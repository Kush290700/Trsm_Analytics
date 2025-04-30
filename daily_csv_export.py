# File: daily_csv_export.py

import pandas as pd
from database import fetch_raw_tables
import datetime, logging, os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("logs/daily_export.log"), logging.StreamHandler()]
)

def export_csv_tables():
    today_str = datetime.datetime.utcnow().strftime('%Y-%m-%d')
    start_date = "2020-01-01"  # Full history

    try:
        raw_data = fetch_raw_tables(start_date, today_str)

        os.makedirs("data", exist_ok=True)
        for name, df in raw_data.items():
            if not df.empty:
                output_path = f"data/{name}.csv"
                df.to_csv(output_path, index=False)
                logging.info(f"✅ Saved {name}.csv ({len(df):,} rows)")
            else:
                logging.warning(f"⚠️ Skipped {name} — no data returned")

    except Exception as e:
        logging.error(f"❌ Failed to export tables: {e}")

if __name__ == "__main__":
    export_csv_tables()
