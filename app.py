import streamlit as st
import pandas as pd
from utils import load_csv_tables, prepare_full_data
from filters import apply_filters
from dashboard_ui import dashboard
from datetime import datetime

st.set_page_config(page_title="TRSM Intelligence", layout="wide")

@st.cache_data
def load_data(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    # 1) Load all raw tables from CSV
    raw = load_csv_tables(csv_dir="data")

    # 2) Prepare & enrich into a single DataFrame
    df = prepare_full_data(raw)

    # 3) Ensure Date is datetime and then apply date slicing
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if start:
        df = df[df["Date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["Date"] <= pd.to_datetime(end)]

    return df

def main():
    st.title("📊 TRSM Advanced Analytics")

    # — Load, prepare & filter —
    df_all = load_data(
        start=min_d.strftime("%Y-%m-%d"),
        end  =max_d.strftime("%Y-%m-%d")
    )
    df = apply_filters(df_all)

    # — Mapping dicts for labels —
    cmap = df_all.set_index("CustomerId")["CustomerName"].to_dict()
    pmap = df_all.set_index("ProductId")["ProductName"].to_dict()

    if df.empty:
        st.warning("⚠️ No data for the selected date range.")
        return

    # — Render the dashboard —
    dashboard(df_all, df, cmap, pmap)

if __name__ == "__main__":
    main()
