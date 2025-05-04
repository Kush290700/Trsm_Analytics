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

    # — Product filter —
    prod_options = ["All"] + [
        f"{sku} – {name}"
        for sku, name in sorted(
            df[['SKU', 'ProductName']]
              .dropna()
              .drop_duplicates()
              .apply(tuple, axis=1)
        )
    ]
    sel_prods = st.sidebar.multiselect(
        "Product(s)", prod_options, default=["All"], key="filt_sku_prod"
    )
    if "All" not in sel_prods:
        chosen = [item.split('–',1)[1].strip() for item in sel_prods]
        mask &= df['ProductName'].isin(chosen)

    # — Region filter —
    regions = ["All"] + get_unique(df, "RegionName")
    sel_regions = st.sidebar.multiselect(
        "Region(s)", regions, default=["All"], key="filt_region"
    )
    if "All" not in sel_regions:
        mask &= df['RegionName'].isin(sel_regions)

    return df.loc[mask].copy()
