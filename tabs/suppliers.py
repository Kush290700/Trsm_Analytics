# File: tabs/suppliers.py
import streamlit as st
import pandas as pd
import plotly.express as px

from utils import (
    filter_by_date,
    get_supplier_summary,
    get_monthly_supplier,
    seasonality_heatmap_data,
    display_seasonality_heatmap,
    fit_prophet
)

@st.cache_data
def cluster_topn(df: pd.DataFrame, total_col: str) -> pd.Series:
    """Perform KMeans clustering on top suppliers."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    X = df[[total_col, "Orders", "MarginPct"]].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=4, random_state=42).fit_predict(X_scaled)
    return pd.Series(labels.astype(str), index=df.index)


def render(df: pd.DataFrame):
    st.subheader("🏭 Supplier Analysis")

    # ─── Sidebar Filters ─────────────────────────────────────────────────
    with st.sidebar.expander("🔧 Suppliers Filters", expanded=True):
        date_range = st.date_input(
            "Date Range", [df.Date.min().date(), df.Date.max().date()], key="sup_date"
        )
        suppliers = ["All"] + sorted(df.SupplierName.dropna().unique())
        sel_sup = st.multiselect("Suppliers", suppliers, default=["All"], key="sup_sel")
        metric = st.selectbox("Metric", ["Revenue", "Cost", "Profit"], index=0, key="sup_metric")
        top_n = st.slider("Top N suppliers", 5, 50, 10, key="sup_topn")
        ma = st.slider("MA window (months)", 1, 12, 3, key="sup_ma")
        hor = st.slider("Forecast horizon (months)", 1, 24, 12, key="sup_hor")
    st.markdown("---")

    # ─── Apply Filters ────────────────────────────────────────────────────
    start_d, end_d = date_range if isinstance(date_range, (list, tuple)) else (date_range, date_range)
    df_f = filter_by_date(df, pd.to_datetime(start_d), pd.to_datetime(end_d))
    if "All" not in sel_sup:
        df_f = df_f[df_f.SupplierName.isin(sel_sup)]
    if df_f.empty:
        st.warning("⚠️ No data for those filters.")
        return

    # ─── Summary & KPIs ───────────────────────────────────────────────────
    col_map = {"Revenue": "TotalRev", "Cost": "TotalCost", "Profit": "TotalProf"}
    total_col = col_map[metric]
    summ = get_supplier_summary(df_f)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Suppliers", f"{summ.SupplierName.nunique():,}")
    c2.metric(f"Total {metric}", f"${summ[total_col].sum():,.0f}")
    c3.metric("Total Profit", f"${summ.TotalProf.sum():,.0f}")
    avg_margin = (summ.TotalProf.sum() / summ.TotalRev.sum() * 100) if summ.TotalRev.sum() else 0
    c4.metric("Avg Margin %", f"{avg_margin:.1f}%")
    c5.metric("Data Points", f"{len(df_f):,}")
    st.markdown("---")

    # ─── Distributions ────────────────────────────────────────────────────
    fig1 = px.histogram(
        summ, x=total_col, nbins=30, marginal="box",
        title=f"{metric} Distribution", labels={total_col: metric}
    )
    fig2 = px.histogram(
        summ, x="MarginPct", nbins=30, marginal="violin",
        title="Margin % Distribution", labels={"MarginPct": "Margin (%)"}
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("---")

    # ─── Top-N Suppliers ──────────────────────────────────────────────────
    topn = summ.nlargest(top_n, total_col)
    fig_top = px.bar(
        topn,
        x=total_col,
        y="SupplierName",
        orientation="h",
        text_auto=",.0f",
        title=f"Top {top_n} Suppliers by {metric}",
        labels={total_col: metric, "SupplierName": "Supplier"}
    )
    st.plotly_chart(fig_top, use_container_width=True)
    st.markdown("---")

    # ─── Interactive Sunburst (lighter hierarchies) ────────────────────────
    with st.expander("🌐 Interactive Sunburst", expanded=False):
        tree_df = (
            df_f.groupby(["RegionName", "CustomerName", "SupplierName", "ProductName"] )[metric]
            .sum().reset_index()
        )
        fig_sb = px.sunburst(
            tree_df,
            path=["RegionName", "CustomerName", "SupplierName", "ProductName"],
            values=metric,
            title=f"{metric} by Region→Customer→Supplier→Product",
            branchvalues="total",
            maxdepth=2  # limit initial depth for performance
        )
        st.plotly_chart(fig_sb, use_container_width=True)
    st.markdown("---")

    # ─── Clustering ───────────────────────────────────────────────────────
    with st.expander("🔍 Clustering of Top Suppliers", expanded=False):
        topn = topn.copy()
        topn["Cluster"] = cluster_topn(topn, total_col)
        fig_cl = px.scatter(
            topn,
            x=total_col,
            y="MarginPct",
            size="Orders",
            color="Cluster",
            hover_name="SupplierName",
            title="Clusters on Top Suppliers"
        )
        st.plotly_chart(fig_cl, use_container_width=True)
    st.markdown("---")

    # ─── Seasonality Heatmap ──────────────────────────────────────────────
    with st.expander("📊 Seasonality Heatmap", expanded=False):
        heat = seasonality_heatmap_data(df_f, "Date", metric)
        display_seasonality_heatmap(heat, f"Seasonality ({metric})", key="sup_season")
    st.markdown("---")

    # ─── Drill-down Table ──────────────────────────────────────────────────
    with st.expander("🔍 Drill-down Table", expanded=False):
        detail = (
            df_f.groupby(["SupplierName", "CustomerName", "ProductName"])  
            .agg(Revenue=("Revenue", "sum"), Profit=("Profit", "sum"), Orders=("OrderId", "nunique"))
            .reset_index()
        )
        st.dataframe(detail, use_container_width=True)
