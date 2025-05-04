# File: tabs/regional.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from datetime import datetime
from utils import (
    filter_by_date,
    seasonality_heatmap_data,
    display_seasonality_heatmap,
    fit_prophet,
    summarize_regions
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def render(df: pd.DataFrame):
    st.subheader("🌎 Regional Analysis")

    # — Data prep —
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()

    # — Sidebar filters & settings —
    with st.sidebar.expander("🔧 Regional Filters & Settings", expanded=True):
        dr = st.date_input(
            "Date Range",
            [df.Date.min().date(), df.Date.max().date()],
            min_value=df.Date.min().date(),
            max_value=df.Date.max().date(),
        )
        metric = st.selectbox("Primary Metric", ["Revenue", "ShippedWeightLb", "Orders", "Profit"])
        gran = st.selectbox("Trend Granularity", ["Monthly", "Quarterly", "Yearly"], index=0)
        freq_map = {"Monthly":"M", "Quarterly":"Q", "Yearly":"Y"}
        freq = freq_map[gran]

        do_fc = (metric=="Revenue") and st.checkbox("Enable Forecast (Revenue only)")
        fc_periods = st.slider("Forecast periods", 1, 24, 12) if do_fc else None

        prods   = ["All"] + sorted(df.ProductName.dropna().unique())
        methods = ["All"] + sorted(df.ShippingMethodName.dropna().unique())
        regs    = ["All"] + sorted(df.RegionName.dropna().unique())

        sel_prods   = st.multiselect("Products", prods, default=["All"])
        sel_methods = st.multiselect("Ship Methods", methods, default=["All"])
        sel_regs    = st.multiselect("Regions", regs, default=["All"])

        st.markdown("---")
        sections = {
            "trend":       True,
            "forecast":    do_fc,
            "yoy":         True,
            "dist":        st.checkbox("Show Distribution Boxplot", False),
            "seasonality": st.checkbox("Show Seasonality Heatmap", True),
            "map":         st.checkbox("Show Geo Map", False),
            "corr":        st.checkbox("Show Correlation Matrix", False),
            "cluster":     st.checkbox("Show Clustering", False),
        }

    # — Apply filters —
    if isinstance(dr, (list, tuple)) and len(dr)==2:
        start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
    else:
        start = end = pd.to_datetime(dr)
    df_f = filter_by_date(df, start, end)
    if "All" not in sel_prods:
        df_f = df_f[df_f.ProductName.isin(sel_prods)]
    if "All" not in sel_methods:
        df_f = df_f[df_f.ShippingMethodName.isin(sel_methods)]
    if "All" not in sel_regs:
        df_f = df_f[df_f.RegionName.isin(sel_regs)]

    if df_f.empty:
        st.warning("No data after filters.")
        return

    # choose column
    col = "OrderId" if metric=="Orders" else metric

    # — KPI cards —
    total_val    = df_f[col].nunique() if metric=="Orders" else df_f[col].sum()
    total_orders = df_f["OrderId"].nunique()
    total_cust   = df_f["CustomerName"].nunique()
    avg_order    = df_f["Revenue"].sum() / total_orders if total_orders else 0
    profit_margin= (df_f["Profit"].sum() / df_f["Revenue"].sum() * 100) if "Profit" in df_f.columns else np.nan

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(f"Total {metric}", f"{total_val:,.0f}")
    k2.metric("Orders", f"{total_orders:,}")
    k3.metric("Customers", f"{total_cust:,}")
    k4.metric("Avg Order $", f"{avg_order:,.0f}")
    k5.metric("Profit Margin", f"{profit_margin:.1f}%")
    st.markdown("---")

    # — Tabs —
    ov, det = st.tabs(["📈 Overview", "🔍 Details"])

    # ─── Overview ────────────────────────────────────────────────────────
    with ov:
        if sections["trend"]:
            st.markdown("### Trend")
            ts = df_f.set_index("Date")[col].resample(freq).sum().reset_index()
            fig_tr = px.line(ts, x="Date", y=col, title=f"{metric} Trend ({gran})")
            st.plotly_chart(fig_tr, use_container_width=True)

            if sections["forecast"] and do_fc:
                st.markdown("### Forecast vs Actual")
                dfp = ts.rename(columns={"Date":"ds",col:"y"})
                fc = fit_prophet(dfp, periods=fc_periods, freq=freq)
                actual = dfp.rename(columns={col:"y","Date":"ds"})
                merged = fc[["ds","yhat"]].merge(actual, on="ds", how="left")
                fig_fc = px.line(merged, x="ds", y=["y","yhat"],
                                 labels={"y":"Actual","yhat":"Forecast"},
                                 title="Prophet Forecast")
                st.plotly_chart(fig_fc, use_container_width=True)

        if sections["yoy"]:
            st.markdown("### Year-over-Year % Δ")
            yoy = (df_f.assign(Year=df_f.Date.dt.year)
                      .groupby(["Year","RegionName"])[col].sum().reset_index())
            yoy["YoY%"] = yoy.groupby("RegionName")[col].pct_change()*100
            latest = yoy.Year.max()
            top_yoy = yoy[yoy.Year==latest].nlargest(10,"YoY%")
            fig_yoy = px.bar(top_yoy, x="RegionName", y="YoY%",
                             text=top_yoy["YoY%"].map("{:+.1f}%".format),
                             title=f"YoY % Δ by Region ({latest})")
            st.plotly_chart(fig_yoy, use_container_width=True)

        if sections["dist"]:
            st.markdown("### Distribution by Region")
            fig_box = px.box(df_f, x="RegionName", y=col, title=f"{metric} Distribution")
            st.plotly_chart(fig_box, use_container_width=True)

        if sections["seasonality"]:
            st.markdown("### Seasonality Heatmap")
            heat = seasonality_heatmap_data(df_f, "Date", col)
            display_seasonality_heatmap(heat, f"Seasonality — {metric}", key="heat_reg")

        if sections["map"] and {"Latitude","Longitude"}.issubset(df_f.columns):
            st.markdown("### Geographic Distribution")
            fig_map = px.scatter_mapbox(
                df_f, lat="Latitude", lon="Longitude",
                size=col, color=col, hover_name="RegionName",
                mapbox_style="open-street-map", zoom=3,
                title=f"{metric} by Location"
            )
            st.plotly_chart(fig_map, use_container_width=True)

        if sections["corr"]:
            st.markdown("### Correlation Matrix")
            numeric = [c for c in ["Revenue","ShippedWeightLb","Profit"] if c in df_f.columns]
            if len(numeric) >= 2:
                corr = df_f[numeric].corr()
                fig_corr = px.imshow(corr, text_auto=True, title="Metric Correlations")
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Not enough numeric columns to compute correlation.")

        if sections["cluster"]:
            st.markdown("### Regional Clustering")
            summ = summarize_regions(df_f, col)
            X = StandardScaler().fit_transform(summ[["Total","Profit"]].fillna(0))
            summ["Cluster"] = KMeans(n_clusters=4, random_state=42).fit_predict(X).astype(str)
            fig_cl = px.scatter(
                summ, x="Total", y="Profit",
                size="Total", color="Cluster", hover_name="RegionName",
                title="Clusters by Total vs Profit"
            )
            st.plotly_chart(fig_cl, use_container_width=True)

        st.markdown("---")
        st.download_button(
            "📥 Download Filtered Data",
            data=df_f.to_csv(index=False),
            file_name="regional_filtered.csv",
            mime="text/csv"
        )

    # ─── Details ────────────────────────────────────────────────────────────────
    with det:
        st.subheader("Region Drill-down")
        choice = st.selectbox("Select Region", regs, index=0)
        sub = df_f if choice=="All" else df_f[df_f.RegionName==choice]

        if sub.empty:
            st.info("No data for this region.")
        else:
            reg_sum = summarize_regions(sub, col)
            st.dataframe(reg_sum, use_container_width=True)

            # side-by-side top products & customers
            p1, p2 = st.columns(2)
            with p1:
                st.markdown("**Top Products**")
                top_p = sub.groupby("ProductName")[col].sum().nlargest(10).reset_index()
                fig_p = px.bar(top_p, x=col, y="ProductName", orientation="h")
                st.plotly_chart(fig_p, use_container_width=True)
            with p2:
                st.markdown("**Top Customers**")
                top_c = sub.groupby("CustomerName")[col].sum().nlargest(10).reset_index()
                fig_c = px.bar(top_c, x=col, y="CustomerName", orientation="h")
                st.plotly_chart(fig_c, use_container_width=True)

        st.download_button(
            "📥 Download Region Summary",
            data=sub.groupby("RegionName")[col].sum().reset_index().to_csv(index=False),
            file_name=f"{choice or 'all'}_region_summary.csv",
            mime="text/csv"
        )
