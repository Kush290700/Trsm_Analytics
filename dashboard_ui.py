# File: dashboard_ui.py
import streamlit as st
import pandas as pd

import tabs.kpis as tab0
import tabs.trend as tab1
import tabs.regional as tab2
import tabs.customers as tab3
import tabs.products as tab4
import tabs.suppliers as tab5


def dashboard(df_all: pd.DataFrame, df: pd.DataFrame, cmap: dict, pmap: dict):
    """
    Renders the main dashboard with all tabs.
    df_all: full data frame
    df: filtered data frame
    cmap: CustomerId → CustomerName map
    pmap: ProductId → ProductName map
    """
    # annotate names for use in tabs
    df = df.copy()
    df_all = df_all.copy()
    df['CustomerName'] = df['CustomerId'].map(cmap).fillna("")
    df['ProductName']  = df['ProductId'].map(pmap).fillna("")
    df_all['CustomerName'] = df_all['CustomerId'].map(cmap).fillna("")
    df_all['ProductName']  = df_all['ProductId'].map(pmap).fillna("")

    tabs = st.tabs(["KPIs","Trend","Regional","Customers","Products","Suppliers"])
    with tabs[0]:
        tab0.render(df_all, df)
    with tabs[1]:
        tab1.render(df)
    with tabs[2]:
        tab2.render(df)
    with tabs[3]:
        tab3.render(df)
    with tabs[4]:
        tab4.render(df)
    with tabs[5]:
        tab5.render(df)
