# tabs/regional.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils import (
    seasonality_heatmap_data,
    display_seasonality_heatmap,
    fit_prophet
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

@st.cache_data
def summarize_regions(df: pd.DataFrame, col: str) -> pd.DataFrame:
    agg = df.groupby("RegionName").agg(
        Total   =(col, "sum"),
        Orders  =("OrderId","nunique"),
        Custs   =("CustomerName","nunique"),
        Profit  =("Profit","sum")
    ).reset_index()
    agg["AvgOrder"] = agg["Total"] / agg["Orders"].replace(0,np.nan)
    agg["MarginPct"] = np.where(agg.Total>0, agg.Profit/agg.Total*100, np.nan)
    return agg

def render(df: pd.DataFrame):
    st.subheader("🌎 Regional Analysis")

    df = df.copy()
    df["Month"] = df.Date.dt.to_period("M").dt.to_timestamp()

    # — Sidebar settings —
    with st.sidebar.expander("🔧 Regional Settings", expanded=True):
        metric = st.selectbox("Metric", ["Revenue","ShippedWeightLb","Orders","Profit"])
        gran   = st.selectbox("Granularity", ["Monthly","Quarterly","Yearly"])
        freq_map = {"Monthly":"M","Quarterly":"Q","Yearly":"Y"}
        freq = freq_map[gran]
        do_fc = st.checkbox("Enable Forecast", metric=="Revenue")
        fc_periods = st.slider("Forecast Periods",1,24,12) if do_fc else None
        methods = ["All"] + sorted(df.ShippingMethodName.dropna().unique())
        sel_methods = st.multiselect("Ship Methods", methods, default=["All"])

    # apply ship-method filter
    if "All" not in sel_methods:
        df = df[df.ShippingMethodName.isin(sel_methods)]
    if df.empty:
        st.warning("No data after filters.")
        return

    col = "OrderId" if metric=="Orders" else metric

    # KPI cards
    total_val   = df[col].nunique() if metric=="Orders" else df[col].sum()
    tot_orders  = df.OrderId.nunique()
    tot_custs   = df.CustomerName.nunique()
    avg_order   = df.Revenue.sum()/tot_orders if tot_orders else 0
    profit_marg = df.Profit.sum()/df.Revenue.sum()*100 if df.Revenue.sum() else np.nan

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric(f"Total {metric}", f"{total_val:,.0f}")
    k2.metric("Orders", f"{tot_orders:,}")
    k3.metric("Customers", f"{tot_custs:,}")
    k4.metric("Avg Order $", f"{avg_order:,.0f}")
    k5.metric("Profit Margin", f"{profit_marg:.1f}%")
    st.markdown("---")

    tabs = st.tabs(["Overview","Details"])
    with tabs[0]:
        # Trend
        ts = df.set_index("Date")[col].resample(freq).sum().reset_index()
        st.markdown("### Trend")
        st.plotly_chart(px.line(ts, x="Date", y=col, title=f"{metric} Trend ({gran})"), use_container_width=True)
        if do_fc:
            st.markdown("### Forecast")
            dfp = ts.rename(columns={"Date":"ds",col:"y"})
            fc = fit_prophet(dfp, periods=fc_periods, freq=freq)
            merged = fc[["ds","yhat"]].merge(ts.rename(columns={col:"y","Date":"ds"}), on="ds", how="left")
            st.plotly_chart(px.line(merged, x="ds", y=["y","yhat"], labels={"y":"Actual","yhat":"Forecast"}), use_container_width=True)

        # YoY % Δ
        st.markdown("### YoY % Δ")
        yoy = df.assign(Year=df.Date.dt.year).groupby(["Year","RegionName"])[col].sum().reset_index()
        yoy["YoY%"] = yoy.groupby("RegionName")[col].pct_change()*100
        latest = yoy.Year.max()
        top_yoy = yoy[yoy.Year==latest].nlargest(10,"YoY%")
        st.plotly_chart(px.bar(top_yoy, x="RegionName", y="YoY%", text=top_yoy["YoY%"].map("{:+.1f}%".format)), use_container_width=True)

        # Distribution
        st.markdown("### Distribution")
        st.plotly_chart(px.box(df, x="RegionName", y=col, title=f"{metric} Distribution"), use_container_width=True)

        # Seasonality
        st.markdown("### Seasonality Heatmap")
        heat = seasonality_heatmap_data(df, "Date", col)
        display_seasonality_heatmap(heat, f"Seasonality — {metric}", key="reg_season")

        # Correlation
        st.markdown("### Correlation")
        nums = ["Revenue","ShippedWeightLb","Profit"]
        corr = df[nums].corr()
        st.plotly_chart(px.imshow(corr, text_auto=True, title="Metric Correlations"), use_container_width=True)

        # Clustering
        st.markdown("### Clustering")
        summ = summarize_regions(df, col)
        X = StandardScaler().fit_transform(summ[["Total","Profit"]].fillna(0))
        summ["Cluster"] = KMeans(n_clusters=4, random_state=42).fit_predict(X).astype(str)
        st.plotly_chart(px.scatter(summ, x="Total", y="Profit", size="Total", color="Cluster", hover_name="RegionName"), use_container_width=True)

    with tabs[1]:
        st.subheader("🔍 Region Drill-down")
        # choose region
        regs = ["All"] + sorted(df.RegionName.dropna().unique())
        choice = st.selectbox("Select Region", regs)
        sub = df if choice=="All" else df[df.RegionName==choice]
        if sub.empty:
            st.info("No data for that region.")
        else:
            reg_sum = summarize_regions(sub, col)
            st.dataframe(reg_sum, use_container_width=True)

            p1,p2 = st.columns(2)
            with p1:
                st.markdown("**Top Products**")
                top_p = sub.groupby("ProductName")[col].sum().nlargest(10).reset_index()
                st.plotly_chart(px.bar(top_p, x=col, y="ProductName", orientation="h"), use_container_width=True)
            with p2:
                st.markdown("**Top Customers**")
                top_c = sub.groupby("CustomerName")[col].sum().nlargest(10).reset_index()
                st.plotly_chart(px.bar(top_c, x=col, y="CustomerName", orientation="h"), use_container_width=True)

            st.download_button("📥 Download Region Summary", data=reg_sum.to_csv(index=False),
                               file_name=f"{choice}_region.csv", mime="text/csv")
