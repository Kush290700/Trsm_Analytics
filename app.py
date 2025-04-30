# File: app.py

import streamlit as st
import pandas as pd
from data_preparation import load_csv_tables, prepare_full_data
from filters import apply_filters
from dashboard_ui import dashboard
from datetime import datetime

st.set_page_config(page_title="TRSM Intelligence", layout="wide")

@st.cache_data(ttl=3600)
def load_data_from_csv():
    raw = load_csv_tables("data")  # data/*.csv
    return prepare_full_data(raw)

def main():
    st.title("📊 TRSM Advanced Analytics")

    # — Load and prepare —
    df_all = load_data_from_csv()

    if df_all.empty:
        st.error("❌ No data available. Please check if your CSVs exist in the /data folder.")
        return

    # — Sidebar date filters —
    min_d = df_all["Date"].min().date()
    max_d = df_all["Date"].max().date()

    start = st.sidebar.date_input("Start Date", value=min_d, min_value=min_d, max_value=max_d)
    end   = st.sidebar.date_input("End Date", value=max_d, min_value=min_d, max_value=max_d)

    df_all = df_all[(df_all["Date"] >= pd.to_datetime(start)) & (df_all["Date"] <= pd.to_datetime(end))]
    df     = apply_filters(df_all)

    # — Mapping dicts for dashboard labels —
    cmap = df_all.set_index("CustomerId")["CustomerName"].to_dict()
    pmap = df_all.set_index("ProductId")["ProductName"].to_dict()

    if df.empty:
        st.warning("⚠️ No data for the selected date range.")
        return

    # — Render dashboard —
    dashboard(df_all, df, cmap, pmap)

if __name__ == "__main__":
    main()
