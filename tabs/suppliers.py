# tabs/suppliers.py
import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events

from utils import (
    filter_by_date,
    get_supplier_summary,
    get_monthly_supplier,
    seasonality_heatmap_data,
    display_seasonality_heatmap,
    fit_prophet,
    compute_volatility
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def render(df: pd.DataFrame):
    st.subheader("🏭 Supplier Analysis")

    # Sidebar
    with st.sidebar.expander("🔧 Suppliers Filters", expanded=True):
        date_range = st.date_input(
            "Date Range",
            [df.Date.min().date(), df.Date.max().date()],
            key="sup_date"
        )
        suppliers = ["All"] + sorted(df.SupplierName.dropna().unique())
        sel_sup = st.multiselect("Suppliers", suppliers, default=["All"], key="sup_sel")
        metric = st.selectbox("Metric", ["Revenue", "Cost", "Profit"], index=0, key="sup_metric")
        top_n = st.slider("Top N suppliers", 5, 50, 10, key="sup_topn")
        ma = st.slider("MA window (months)", 1, 12, 3, key="sup_ma")
        hor = st.slider("Forecast horizon (months)", 1, 24, 12, key="sup_hor")

    # Date filter
    start_d, end_d = date_range if isinstance(date_range, (list, tuple)) else (date_range, date_range)
    df_f = filter_by_date(df, pd.to_datetime(start_d), pd.to_datetime(end_d))

    # Supplier filter
    if "All" not in sel_sup:
        df_f = df_f[df_f.SupplierName.isin(sel_sup)]

    if df_f.empty:
        st.warning("⚠️ No data for those filters.")
        return

    # Map metric → summary column
    col_map = {"Revenue": "TotalRev", "Cost": "TotalCost", "Profit": "TotalProf"}
    total_col = col_map[metric]

    # KPI cards
    summ = get_supplier_summary(df_f)
    total_sup  = summ.SupplierName.nunique()
    total_met  = summ[total_col].sum()
    total_prof = summ.TotalProf.sum()
    avg_margin = (total_prof / summ.TotalRev.sum() * 100) if summ.TotalRev.sum() else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Suppliers", f"{total_sup:,}")
    c2.metric(f"Total {metric}", f"${total_met:,.0f}")
    c3.metric("Total Profit", f"${total_prof:,.0f}")
    c4.metric("Avg Margin %", f"{avg_margin:.1f}%")
    c5.metric("Data Points", f"{len(df_f):,}")
    st.markdown("---")

    # Distribution histograms
    fig1 = px.histogram(
        summ,
        x=total_col,
        nbins=30,
        marginal="box",
        title=f"{metric} Distribution",
        labels={total_col: metric}
    )
    fig2 = px.histogram(
        summ,
        x="MarginPct",
        nbins=30,
        marginal="violin",
        title="Margin % Distribution",
        labels={"MarginPct": "Margin (%)"}
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("---")

    # Top-N bar
    topn = summ.nlargest(top_n, total_col)
    fig_top = px.bar(
        topn,
        x=total_col, y="SupplierName",
        orientation="h",
        text_auto=",.0f",
        title=f"Top {top_n} Suppliers by {metric}",
        labels={total_col: metric, "SupplierName": "Supplier"}
    )
    st.plotly_chart(fig_top, use_container_width=True)
    st.markdown("---")

    # Time-series + MA + Forecast
    ts = get_monthly_supplier(df_f, metric)
    ts["MA"] = ts[metric].rolling(ma).mean()
    fig_ts = px.line(
        ts,
        x="Date", y=[metric, "MA"],
        title=f"{metric} Trend (MA={ma}mo)"
    )
    fig_ts.update_traces(selector={"name": "MA"}, line_dash="dash")
    st.plotly_chart(fig_ts, use_container_width=True)

    if len(ts) >= 2:
        dfp = ts.rename(columns={"Date": "ds", metric: "y"})[["ds", "y"]]
        fc = fit_prophet(dfp, periods=hor, freq="M")
        fig_fc = px.line(fc, x="ds", y="yhat", title=f"{metric} Forecast (+{hor}mo)")
        fig_fc.add_scatter(x=fc.ds, y=fc.yhat_upper, mode="lines", line_dash="dash", name="Upper")
        fig_fc.add_scatter(x=fc.ds, y=fc.yhat_lower, mode="lines", line_dash="dash", name="Lower")
        st.plotly_chart(fig_fc, use_container_width=True)
    st.markdown("---")

    # Treemaps
    for path, title in [
        (["RegionName", "SupplierName", "ProductName"],  f"{metric} by Region→Supplier→Product"),
        (["RegionName", "SupplierName", "CustomerName"], f"{metric} by Region→Supplier→Customer")
    ]:
        treedf = df_f.groupby(path)[metric].sum().reset_index()
        fig_tm = px.treemap(treedf, path=path, values=metric, title=title)
        st.plotly_chart(fig_tm, use_container_width=True)
        st.markdown("---")

    # Scatter & drill-down
    fig_sc = px.scatter(
        summ,
        x=total_col, y="MarginPct", size="Orders",
        hover_name="SupplierName",
        title=f"{metric} vs Margin %"
    )
    clicked = plotly_events(fig_sc, click_event=True)
    st.plotly_chart(fig_sc, use_container_width=True)

    if clicked:
        sup = clicked[0].get("hovertext") or topn.iloc[clicked[0]["pointIndex"]]["SupplierName"]
        st.markdown(f"#### Details for **{sup}**")
        df_sup = df_f[df_f.SupplierName == sup]
        prod = (
            df_sup
            .groupby("ProductName")
            .agg(
                Revenue=("Revenue", "sum"),
                Cost=("Cost",       "sum"),
                Profit=("Profit",   "sum"),
                Orders=("OrderId",  "nunique")
            )
            .reset_index()
            .sort_values("Revenue", ascending=False)
        )
        st.dataframe(prod, use_container_width=True)
        st.markdown("---")

    # Volatility by supplier
    vol = (
        compute_volatility(
            df_f,
            metric,
            freq="M",
            group_col="SupplierName"
        )
        .dropna(subset=["mean", "std", "CV"])
        .query("mean > 0")
    )
    st.plotly_chart(
        px.scatter(
            vol,
            x="mean", y="CV", size="std",
            hover_name="SupplierName",
            title=f"{metric} Volatility (Mean vs CV)",
            labels={"mean": f"Avg {metric}", "CV": "Coefficient of Variation", "std": "Std Dev"}
        ),
        use_container_width=True
    )
    st.markdown("---")

    # K-Means clustering on Top-N
    X = StandardScaler().fit_transform(topn[[total_col, "Orders", "MarginPct"]].fillna(0))
    topn["Cluster"] = KMeans(n_clusters=4, random_state=42).fit_predict(X).astype(str)
    fig_cl = px.scatter(
        topn,
        x=total_col, y="MarginPct", size="Orders",
        color="Cluster", hover_name="SupplierName",
        title="Clusters on Top Suppliers"
    )
    st.plotly_chart(fig_cl, use_container_width=True)
    st.markdown("---")

    # Seasonality heatmap
    heat = seasonality_heatmap_data(df_f, "Date", metric)
    display_seasonality_heatmap(heat, f"Seasonality ({metric})", key="sup_season")
    st.markdown("---")

    # Drill-down table
    detail = (
        df_f
        .groupby(["SupplierName", "CustomerName", "ProductName"])
        .agg(Revenue=("Revenue", "sum"), Profit=("Profit", "sum"), Orders=("OrderId", "nunique"))
        .reset_index()
    )
    st.subheader("🔍 Drill-down Table")
    st.dataframe(detail, use_container_width=True)
