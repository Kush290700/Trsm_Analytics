# File: app.py
import streamlit as st
import pandas as pd
import gc
from datetime import datetime

from utils import load_csv_tables, prepare_full_data
from filters import apply_filters
from dashboard_ui import dashboard

st.set_page_config(page_title="TRSM Intelligence", layout="wide")

@st.cache_data(show_spinner=False)
def load_full_data() -> pd.DataFrame:
    """
    Load and enrich all CSV tables exactly once per session.
    Returns a single DataFrame with a proper Date column.
    """
    raw = load_csv_tables(csv_dir="data")
    df = prepare_full_data(raw)
    # Normalize and parse Date once
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

def main():
    st.title("📊 TRSM Advanced Analytics")

    # — Sidebar date filters (default from 2021) —
    min_d = st.sidebar.date_input("Start Date", value=datetime(2021, 1, 1))
    max_d = st.sidebar.date_input("End Date",   value=datetime.today())

    # — Load the full data once, then slice cheaply in memory —
    df_full = load_full_data()
    mask = (df_full["Date"] >= pd.to_datetime(min_d)) & (df_full["Date"] <= pd.to_datetime(max_d))
    df_all = df_full.loc[mask]

    # Force a garbage‐collection pass after slicing to free any dropped objects
    gc.collect()

    # — Apply your product/region filters (also cached cheaply) —
    df = apply_filters(df_all)

    # — Build lookup maps for the dashboard —
    cmap = df_all.set_index("CustomerId")["CustomerName"].to_dict()
    pmap = df_all.set_index("ProductId") ["ProductName"].to_dict()

    if df.empty:
        st.warning("⚠️ No data for the selected date range.")
        return

    # — Render all of your tabs/charts —
    dashboard(df_all, df, cmap, pmap)

if __name__ == "__main__":
    main()
