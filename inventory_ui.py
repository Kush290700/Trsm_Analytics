# inventory_ui.py

import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from holding_cost import compute_holding_cost

# ──────────────────────────────────────────────────────────────────────────────
# LOCAL UTILITIES (so we don’t clash with dashboard’s utils.py)
# ──────────────────────────────────────────────────────────────────────────────

def clean_numeric_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Remove any non‑numeric characters and coerce to float, filling NaN with 0.
    """
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace("", "0")
        .astype(float)
    )
    return df

# ──────────────────────────────────────────────────────────────────────────────
# MAIN INVENTORY DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────

def run_inventory_dashboard():
    st.header("🏭 Advanced Inventory & Holding Cost Dashboard")

    uploaded_file = st.file_uploader("Upload Inventory CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Please upload a CSV file to begin.")
        return

    # --- read & clean ---
    df = pd.read_csv(uploaded_file)
    df = clean_numeric_column(df, 'Cost_pr')
    df = clean_numeric_column(df, 'WeightLb')
    df['OriginDate'] = pd.to_datetime(df['OriginDate'], errors='coerce')

    # --- compute holding costs (may raise) ---
    try:
        df = compute_holding_cost(df)
    except ValueError as e:
        st.error(str(e))
        return

    # --- sidebar filters ---
    st.sidebar.header("Inventory Filters")
    state_options = sorted(df["State"].dropna().unique())
    selected_states = st.sidebar.multiselect("Select State(s)", options=state_options, default=state_options)

    prod_search = st.sidebar.text_input("Search Product", value="")
    all_products = sorted(df["Product"].dropna().unique())
    if prod_search:
        filtered_products = [p for p in all_products if prod_search.lower() in p.lower()]
    else:
        filtered_products = all_products
    selected_products = st.sidebar.multiselect(
        "Select Product(s)",
        options=filtered_products,
        default=filtered_products
    )

    # --- apply filters & prepare ---
    filtered_df = df[
        (df["State"].isin(selected_states)) &
        (df["Product"].isin(selected_products))
    ].copy()
    filtered_df["SKU_Product"] = filtered_df["SKU"].astype(str) + " - " + filtered_df["Product"]

    # --- summary metrics ---
    st.subheader("Inventory Summary Metrics")
    total_skus         = filtered_df['SKU'].nunique()
    total_inventory    = filtered_df['ItemCount'].sum()
    total_inv_value    = filtered_df['InventoryValue'].sum()
    total_holding_cost = filtered_df['TotalHoldingCost'].sum()
    avg_holding_pct    = filtered_df['HoldingCostPercent'].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Unique SKUs", total_skus)
    c2.metric("Total Items", total_inventory)
    c3.metric("Inventory Value", f"${total_inv_value:,.2f}")
    c4.metric("Holding Cost", f"${total_holding_cost:,.2f}")
    c5.metric("Avg Holding %", f"{avg_holding_pct:.2f}%")

    # --- raw table ---
    st.subheader("Detailed Inventory Data")
    st.dataframe(filtered_df)

    # --- primary charts ---
    st.subheader("Primary Interactive Visualizations")
    base_color = alt.Scale(scheme='tableau10')

    # 1) Avg Holding % by State
    state_sel = alt.selection_point(fields=["State"], bind="legend")
    chart1 = (
        alt.Chart(filtered_df)
        .mark_bar()
        .encode(
            x=alt.X('State:N', title="State", sort='-y'),
            y=alt.Y('mean(HoldingCostPercent):Q', title="Avg Holding Cost %"),
            color=alt.Color('State:N', scale=base_color),
            opacity=alt.condition(state_sel, alt.value(1), alt.value(0.3)),
            tooltip=[
                alt.Tooltip('State:N', title="State"),
                alt.Tooltip('mean(HoldingCostPercent):Q', title="Avg Holding Cost %", format=".2f"),
                alt.Tooltip('count():Q', title="Count")
            ]
        )
        .add_params(state_sel)
        .properties(width=600, height=400, title="Average Holding Cost % by State")
        .interactive()
    )
    st.altair_chart(chart1, use_container_width=True)

    # 2) Total Holding Cost by Product
    prod_sel = alt.selection_point(fields=["Product"], bind="legend")
    chart2 = (
        alt.Chart(filtered_df)
        .mark_bar()
        .encode(
            x=alt.X('Product:N', title="Product", sort='-y'),
            y=alt.Y('sum(TotalHoldingCost):Q', title="Total Holding Cost"),
            color=alt.Color('Product:N', scale=base_color),
            opacity=alt.condition(prod_sel, alt.value(1), alt.value(0.3)),
            tooltip=[
                alt.Tooltip('Product:N', title="Product"),
                alt.Tooltip('sum(TotalHoldingCost):Q', title="Holding Cost", format="$.2f"),
                alt.Tooltip('sum(ItemCount):Q', title="Total Items")
            ]
        )
        .add_params(prod_sel)
        .properties(width=600, height=400, title="Total Holding Cost by Product")
        .interactive()
    )
    st.altair_chart(chart2, use_container_width=True)

    # ... (the rest of your charts remain exactly as before) ...

    # final bubble chart needs a brush:
    st.subheader("Advanced Inventory Overview")
    brush = alt.selection_interval()
    chart_overview = (
        alt.Chart(filtered_df)
        .mark_circle()
        .encode(
            x=alt.X("DaysInStorage:Q", title="Days In Storage"),
            y=alt.Y("HoldingCostPercent:Q", title="Holding Cost %"),
            size=alt.Size("InventoryValue:Q", title="Inventory Value"),
            color=alt.Color("Product:N", legend=alt.Legend(title="Product")),
            tooltip=[
                alt.Tooltip("SKU:N", title="SKU"),
                alt.Tooltip("Product:N", title="Product"),
                alt.Tooltip("DaysInStorage:Q", title="Days In Storage"),
                alt.Tooltip("HoldingCostPercent:Q", title="Holding Cost %", format=".2f"),
                alt.Tooltip("InventoryValue:Q", title="Inventory Value", format="$.2f")
            ]
        )
        .add_params(brush)
        .properties(width=800, height=500, title="Inventory Overview")
        .interactive()
    )
    st.altair_chart(chart_overview, use_container_width=True)
