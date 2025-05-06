# File: app.py
import streamlit as st
import pandas as pd
from utils import load_csv_tables, prepare_full_data
from filters import apply_filters
from dashboard_ui import dashboard
from datetime import datetime

st.set_page_config(page_title="TRSM Intelligence", layout="wide")

@st.cache_data(show_spinner=False)
def load_full_data() -> pd.DataFrame:
    """Load & enrich the raw CSVs into one DataFrame (no date filtering)."""
    raw = load_csv_tables(csv_dir="data")
    df = prepare_full_data(raw)
    # ensure Date is datetime once
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

def main():
    st.title("📊 TRSM Advanced Analytics")

    # — Sidebar date filters (default from 2021) —
    min_d = st.sidebar.date_input("Start Date", value=datetime(2021, 1, 1))
    max_d = st.sidebar.date_input("End Date",   value=datetime.today())

    # — Load full data once, then slice in‐memory —
    df_full = load_full_data()
    # slice cheaply in‐memory
    mask = (df_full["Date"] >= pd.to_datetime(min_d)) & (df_full["Date"] <= pd.to_datetime(max_d))
    df_all = df_full.loc[mask]

    # — Apply your SKU/Region filters —
    df = apply_filters(df_all)

    # — Build lookup maps for the dashboard —
    cmap = df_all.set_index("CustomerId")["CustomerName"].to_dict()
    pmap = df_all.set_index("ProductId") ["ProductName"].to_dict()

    if df.empty:
        st.warning("⚠️ No data for the selected date range.")
        return

    # — Render the dashboard —
    dashboard(df_all, df, cmap, pmap)

if __name__ == "__main__":
    main()
