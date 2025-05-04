# File: dashboard_ui.py
import streamlit as st
import pandas as pd
from utils import (
    fit_prophet, seasonality_heatmap_data, display_seasonality_heatmap,
    rfm_scatter, get_supplier_summary, get_monthly_supplier, compute_interpurchase, compute_volatility
)
import tabs.kpis as tab0, tabs.trend as tab1, tabs.regional as tab2
import tabs.customers as tab3, tabs.products as tab4, tabs.suppliers as tab5

@st.cache_data
def build_maps(df_all: pd.DataFrame):
    cmap = df_all.set_index("CustomerId")["CustomerName"].to_dict()
    pmap = df_all.set_index("ProductId")["ProductName"].to_dict()
    return cmap, pmap


def dashboard(df_all: pd.DataFrame, df: pd.DataFrame):
    cmap, pmap = build_maps(df_all)
    df["CustomerName"] = df.CustomerId.map(cmap).astype("category")
    df["ProductName"]  = df.ProductId.map(pmap).astype("category")

    tabs = st.tabs(["KPIs","Trend","Regional","Customers","Products","Suppliers"])
    with tabs[0]: tab0.render(df_all, df)
    with tabs[1]: tab1.render(df)
    with tabs[2]: tab2.render(df)
    with tabs[3]: tab3.render(df)
    with tabs[4]: tab4.render(df)
    with tabs[5]: tab5.render(df)
