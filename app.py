
# File: app.py
import streamlit as st
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

    df_all = load_data_from_csv()
    if df_all.empty:
        st.error("❌ No data available. Please check if your CSVs exist in the /data folder.")
        return

    # Apply global filters (date, product, region)
    df = apply_filters(df_all)
    if df.empty:
        st.warning("⚠️ No data for the selected filters.")
        return

    # Render the dashboard
    dashboard(df_all, df)

if __name__ == "__main__":
    main()
