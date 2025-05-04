# tabs/products.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils import seasonality_heatmap_data, display_seasonality_heatmap, fit_prophet, compute_volatility

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

@st.cache_data
def summarize_products(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("ProductName", as_index=False).agg(
        Revenue=("Revenue","sum"),
        Units  =("ItemCount","sum"),
        Profit =("Profit","sum")
    )

def render(df: pd.DataFrame):
    st.subheader("📦 Product Analytics")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["Date"], inplace=True)

    # — Sidebar settings only —
    with st.sidebar.expander("🔧 Product Settings", expanded=True):
        metric   = st.selectbox("Primary Metric", ["Revenue","Units","Profit"], key="prod_metric")
        top_n    = st.slider("Top N Products", 5, 50, 10, key="prod_topn")
        ma_window= st.slider("MA Window (months)", 1, 12, 3, key="prod_ma")
        horizon  = st.slider("Forecast Horizon (months)", 1, 24, 12, key="prod_hor")

    summary = summarize_products(df)
    total_rev  = summary.Revenue.sum()
    total_prof = summary.Profit.sum()
    avg_margin = (total_prof/total_rev*100) if total_rev else 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Products", f"{summary.shape[0]:,}")
    c2.metric("SKUs",     f"{df.SKU.nunique():,}")
    c3.metric("Revenue",  f"${total_rev:,.0f}")
    c4.metric("Avg Margin %", f"{avg_margin:.1f}%")
    st.markdown("---")

    topn = summary.nlargest(top_n, metric)
    col1,col2 = st.columns(2)
    col1.plotly_chart(px.bar(topn, x="Revenue", y="ProductName", orientation="h", text_auto=",.0f"), use_container_width=True)
    col2.plotly_chart(px.bar(topn, x="Units",   y="ProductName", orientation="h", text_auto=",.0f"), use_container_width=True)
    st.markdown("---")

    fig_dist = px.histogram(summary, x=metric, nbins=30, marginal="box", title=f"{metric} Distribution")
    st.plotly_chart(fig_dist, use_container_width=True)
    st.markdown("---")

    ts = df.set_index("Date")[metric].resample("M").sum().reset_index(name=metric)
    ts["MA"] = ts[metric].rolling(ma_window).mean()

    fig_ts = px.line(ts, x="Date", y=[metric,"MA"], labels={"value":metric}, title=f"{metric} Trend")
    fig_ts.update_traces(selector={"name":"MA"}, line_dash="dash")
    st.plotly_chart(fig_ts, use_container_width=True)

    if metric=="Revenue" and len(ts)>=2:
        fc = fit_prophet(ts.rename(columns={"Date":"ds",metric:"y"})[["ds","y"]], periods=horizon, freq="M")
        fig_fc = px.line(fc, x="ds", y="yhat", title="Forecast")
        fig_fc.add_scatter(x=fc.ds, y=fc.yhat_upper, mode="lines", line_dash="dash", name="Upper")
        fig_fc.add_scatter(x=fc.ds, y=fc.yhat_lower, mode="lines", line_dash="dash", name="Lower")
        st.plotly_chart(fig_fc, use_container_width=True)
    st.markdown("---")

    heat = seasonality_heatmap_data(df, "Date", metric)
    display_seasonality_heatmap(heat, f"Seasonality ({metric})", key="prod_season")
    st.markdown("---")

    abc = topn.assign(CumPct=lambda d: d[metric].cumsum()/d[metric].sum()*100)
    abc["Class"] = pd.cut(abc.CumPct, [0,80,95,100], labels=["A","B","C"])
    a_col,b_col = st.columns(2)
    a_col.plotly_chart(px.bar(abc, x="ProductName", y="CumPct", color="Class", title="ABC Class"), use_container_width=True)
    share = abc.groupby("Class", as_index=False)[metric].sum()
    b_col.plotly_chart(px.pie(share, names="Class", values=metric, hole=0.4, title="Share by Class"), use_container_width=True)
    st.markdown("---")

    vol_df = summary.assign(MarginPct=summary.Profit/summary.Revenue*100)
    st.plotly_chart(px.scatter(vol_df, x="Revenue", y="MarginPct", size="Units", color="MarginPct", title="Revenue vs Margin %"), use_container_width=True)
    st.markdown("---")

    vol_stats = compute_volatility(df, metric, period="M")
    st.plotly_chart(px.scatter(vol_stats, x="mean", y="CV", size="std", hover_name="ProductName", title="Volatility"), use_container_width=True)
    st.markdown("---")

    X = StandardScaler().fit_transform(topn[[metric,"Units","Profit"]].fillna(0))
    topn["Cluster"] = KMeans(n_clusters=4, random_state=42).fit_predict(X).astype(str)
    st.plotly_chart(px.scatter(topn, x="Revenue", y="Units", size="Profit", color="Cluster", hover_data=["ProductName"], title="Clusters"), use_container_width=True)
    st.markdown("---")
    st.download_button("📥 Download Product Data", data=summary.to_csv(index=False), file_name="products.csv", mime="text/csv")
