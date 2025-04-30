# File: export_to_excel.py

import datetime
import pandas as pd

from database import fetch_raw_tables
from data_preparation import prepare_full_data

def build_excel(start_date: str, end_date: str, out_path: str):
    # 1) load & prep
    raw = fetch_raw_tables(start_date, end_date)
    df  = prepare_full_data(raw)

    # 2) open a writer
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        workbook  = writer.book

        # --- Sheet 1: Raw data dump ---
        df.to_excel(writer, sheet_name="Data", index=False)

        # --- Sheet 2: Time series of daily revenue ---
        daily = (
            df.groupby("Date", as_index=False)
              .agg(Revenue=("Revenue", "sum"))
        )
        daily.to_excel(writer, sheet_name="DailyRev", index=False)
        ws = writer.sheets["DailyRev"]
        chart = workbook.add_chart({"type": "line"})
        chart.add_series({
            "name":       "Revenue",
            "categories": ["DailyRev", 1, 0, len(daily), 0],
            "values":     ["DailyRev", 1, 1, len(daily), 1],
        })
        chart.set_title({"name": "Daily Revenue"})
        chart.set_x_axis({"name": "Date"})
        chart.set_y_axis({"name": "Revenue"})
        ws.insert_chart("D2", chart, {"x_scale": 1.5, "y_scale": 1.5})

        # --- Sheet 3: RFM pie chart ---
        # (Replicate your Streamlit "segments" chart)
        seg_counts = (
            df.assign(RFM=df["CustomerId"])  # if you actually have an RFM col, adjust
              .groupby("RFM")
              .size()
              .reset_index(name="Count")
        )
        seg_counts.to_excel(writer, sheet_name="Segments", index=False)
        ws2 = writer.sheets["Segments"]
        pie = workbook.add_chart({"type": "pie"})
        pie.add_series({
            "name":       "Customer segments",
            "categories": ["Segments", 1, 0, len(seg_counts), 0],
            "values":     ["Segments", 1, 1, len(seg_counts), 1],
        })
        pie.set_title({"name": "Customer Segments"})
        ws2.insert_chart("D2", pie, {"x_scale": 1.2, "y_scale": 1.2})

        # --- Sheet 4: Top products bar chart ---
        top_prods = (
            df.groupby("ProductName", as_index=False)
              .agg(Revenue=("Revenue", "sum"))
              .sort_values("Revenue", ascending=False)
              .head(10)
        )
        top_prods.to_excel(writer, sheet_name="TopProducts", index=False)
        ws3 = writer.sheets["TopProducts"]
        bar = workbook.add_chart({"type": "column"})
        bar.add_series({
            "name":       "Revenue",
            "categories": ["TopProducts", 1, 0, 10, 0],
            "values":     ["TopProducts", 1, 1, 10, 1],
        })
        bar.set_title({"name": "Top 10 Products by Revenue"})
        bar.set_x_axis({"name": "Product"})
        bar.set_y_axis({"name": "Revenue"})
        ws3.insert_chart("D2", bar, {"x_scale": 1.5, "y_scale": 1.2})

        # you can add more sheets & charts here following the same pattern

    print(f"✅ Wrote {out_path}")

if __name__ == "__main__":
    # default to last year → today
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=365)
    build_excel(start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
                out_path="TRSM_report.xlsx")
