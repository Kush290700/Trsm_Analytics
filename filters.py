# File: filters.py

import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def get_unique(df: pd.DataFrame, col: str) -> list:
    """Return sorted unique non-null values for a given column."""
    if col in df.columns:
        return sorted(df[col].dropna().unique().tolist())
    return []

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("🔎 Filters")

    # — Date range picker —
    dmin, dmax = df['Date'].min().date(), df['Date'].max().date()
    date_selection = st.sidebar.date_input(
        "Date Range",
        [dmin, dmax],
        min_value=dmin,
        max_value=dmax,
        key="filt_date"
    )
    if isinstance(date_selection, (list, tuple)) and len(date_selection) == 2:
        start_date, end_date = date_selection
    else:
        start_date = end_date = date_selection

    mask = df['Date'].dt.date.between(start_date, end_date)

    # — Product (SKU – ProductName) filter —
    sku_prod = (
        df[['SKU', 'ProductName']]
        .dropna(subset=['SKU','ProductName'])
        .drop_duplicates()
        .sort_values(['SKU','ProductName'])
    )
    prod_options = ["All"] + [
        f"{row.SKU} – {row.ProductName}"
        for _, row in sku_prod.iterrows()
    ]
    selected_prods = st.sidebar.multiselect(
        "Product(s)",
        prod_options,
        default=["All"],
        key="filt_sku_prod"
    )
    if "All" not in selected_prods:
        chosen_names = [item.split("–",1)[1].strip() for item in selected_prods]
        mask &= df['ProductName'].isin(chosen_names)

    # — Region filter —
    regions = get_unique(df, "RegionName")
    sel_regions = st.sidebar.multiselect(
        "Region(s)",
        ["All"] + regions,
        default=["All"],
        key="filt_region"
    )
    if "All" not in sel_regions:
        mask &= df['RegionName'].isin(sel_regions)

    # — Sales Rep filter —
    reps = get_unique(df, "SalesRepId")
    sel_reps = st.sidebar.multiselect(
        "Sales Rep(s)",
        ["All"] + reps,
        default=["All"],
        key="filt_rep"
    )
    if "All" not in sel_reps:
        mask &= df['SalesRepId'].isin(sel_reps)

    return df.loc[mask].copy()
