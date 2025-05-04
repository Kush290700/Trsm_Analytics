# app.py

import streamlit as st
import pandas as pd
from data_preparation import load_csv_tables, prepare_full_data
from filters import apply_filters
from dashboard_ui import dashboard

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
        st.error("❌ No data available. Please check your CSVs in `/data`.")
        return

    # — Global date picker (only here) —
    dmin, dmax = df_all["Date"].min().date(), df_all["Date"].max().date()
    start = st.sidebar.date_input("Start Date", value=dmin, min_value=dmin, max_value=dmax)
    end   = st.sidebar.date_input("End Date",   value=dmax, min_value=dmin, max_value=dmax)

    # apply date range
    df_time = df_all.loc[
        (df_all["Date"] >= pd.to_datetime(start)) &
        (df_all["Date"] <= pd.to_datetime(end))
    ]

    # — Attribute filters —
    df = apply_filters(df_time)
    if df.empty:
        st.warning("⚠️ No data after filters.")
        return

    # — Build mapping dicts once —
    cmap = df_all.set_index("CustomerId")["CustomerName"].to_dict()
    pmap = df_all.set_index("ProductId")  ["ProductName"].to_dict()

    # — Render all tabs —
    dashboard(df_all, df, cmap, pmap)

if __name__ == "__main__":
    main()
