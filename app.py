# app.py
#!/usr/bin/env python3
"""
Streamlit app that loads data directly from Snowflake, prepares the full analytics DataFrame,
and displays an interactive TRSM Intelligence dashboard.

Prerequisites:
  - Streamlit installed:
      pip install streamlit snowflake-connector-python pandas numpy python-dotenv
  - .env file adjacent to this script containing:
      SF_USER, SF_PWD, SF_ACCOUNT, SF_WAREHOUSE, SF_DATABASE, SF_SCHEMA

Launch:
  streamlit run app.py
"""
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import snowflake.connector

from data_preparation import prepare_full_data
from filters import apply_filters
from dashboard_ui import dashboard

# ─── Load environment variables ─────────────────────────────────────────────
load_dotenv()

# ─── Streamlit page config ─────────────────────────────────────────────────
st.set_page_config(page_title="TRSM Intelligence", layout="wide")

# ─── Snowflake connection factory ──────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_snowflake_conn():
    return snowflake.connector.connect(
        user=os.getenv("SF_USER"),
        password=os.getenv("SF_PWD"),
        account=os.getenv("SF_ACCOUNT"),
        warehouse=os.getenv("SF_WAREHOUSE"),
        database=os.getenv("SF_DATABASE"),
        schema=os.getenv("SF_SCHEMA")
    )

# ─── Load & prepare data from Snowflake ───────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    conn = get_snowflake_conn()
    # table names in Snowflake (uppercase)
    tables = [
        "ORDERS", "ORDER_LINES", "PRODUCTS", "CUSTOMERS",
        "REGIONS", "SHIPPERS", "SUPPLIERS", "SHIPPING_METHODS", "PACKS"
    ]
    raw = {}
    for tbl in tables:
        key = tbl.lower()
        raw[key] = pd.read_sql(f"SELECT * FROM {tbl}", conn)
    # merge & compute analytics fields
    df_full = prepare_full_data(raw)
    return df_full

# ─── Main application ──────────────────────────────────────────────────────
def main():
    st.title("📊 TRSM Advanced Analytics")

    # Load data
    with st.spinner("Loading data from Snowflake..."):
        df_all = load_data()

    if df_all.empty:
        st.error("❌ No data available from Snowflake.")
        return

    # Sidebar date range filter
    dmin, dmax = df_all["Date"].min().date(), df_all["Date"].max().date()
    start = st.sidebar.date_input("Start Date", value=dmin, min_value=dmin, max_value=dmax)
    end   = st.sidebar.date_input("End Date",   value=dmax, min_value=dmin, max_value=dmax)

    df_time = df_all[(df_all["Date"] >= pd.to_datetime(start)) & (df_all["Date"] <= pd.to_datetime(end))]

    # Additional attribute filters
    df_filtered = apply_filters(df_time)
    if df_filtered.empty:
        st.warning("⚠️ No data after applying filters.")
        return

    # Lookup maps for UI labels
    cmap = df_all.set_index("CustomerId")["CustomerName"].to_dict()
    pmap = df_all.set_index("ProductId")["ProductName"].to_dict()

    # Render dashboard tabs
    dashboard(df_all, df_filtered, cmap, pmap)

if __name__ == "__main__":
    main()
