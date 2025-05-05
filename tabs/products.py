# File: tabs/products.py
import streamlit as st
import pandas as pd
import plotly.express as px

from utils import (
    seasonality_heatmap_data,
    display_seasonality_heatmap,
    fit_prophet
)

@st.cache_data
def summarize_products(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by product."""
    return (
        df.groupby("ProductName", as_index=False)
          .agg(
              Revenue=("Revenue", "sum"),
              Units=("ItemCount", "sum"),
              Profit=("Profit", "sum")
          )
    )

@st.cache_data
def cluster_products(df: pd.DataFrame, metric: str) -> pd.Series:
    """Perform KMeans clustering on products based on metric, Units, Profit."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    X = df[[metric, "Units", "Profit"]].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=4, random_state=42).fit_predict(X_scaled)
    return pd.Series(labels.astype(str), index=df.index)


def render(df: pd.DataFrame):
    st.subheader("📦 Product Analytics")

    # Ensure Date is datetime once
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Sidebar settings
    with st.sidebar.expander("🔧 Product Settings", expanded=True):
        metric    = st.selectbox("Primary Metric", ["Revenue","Units","Profit"], key="prod_metric")
        top_n     = st.slider("Top N Products", 5, 50, 10, key="prod_topn")
        ma_window = st.slider("MA Window (months)", 1, 12, 3, key="prod_ma")
        horizon   = st.slider("Forecast Horizon (months)", 1, 24, 12, key="prod_hor")
    st.markdown("---")

    # Summary & KPIs
    summary = summarize_products(df)
    total_rev  = summary.Revenue.sum()
    total_prof = summary.Profit.sum()
    avg_margin = (total_prof / total_rev * 100) if total_rev else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products", f"{summary.shape[0]:,}")
    if 'SKU' in df.columns:
        c2.metric("SKUs", f"{df.SKU.nunique():,}")
    else:
        c2.write("")
    c3.metric("Revenue", f"${total_rev:,.0f}")
    c4.metric("Avg Margin %", f"{avg_margin:.1f}%")
    st.markdown("---")

    # Top-N & Distribution
    with st.expander("🔝 Top-N & Distribution", expanded=False):
        topn = summary.nlargest(top_n, metric)
        col1, col2 = st.columns(2)
        col1.plotly_chart(
            px.bar(topn, x="Revenue",  y="ProductName", orientation="h",
                   text_auto=",.0f", title=f"Top {top_n} by Revenue"),
            use_container_width=True
        )
        col2.plotly_chart(
            px.bar(topn, x="Units",    y="ProductName", orientation="h",
                   text_auto=",.0f", title=f"Top {top_n} by Units"),
            use_container_width=True
        )
        st.markdown("---")
        st.plotly_chart(
            px.histogram(summary, x=metric, nbins=30, marginal="box",
                         title=f"{metric} Distribution"),
            use_container_width=True
        )
    st.markdown("---")

    # Time Series & Forecast
    with st.expander("📈 Trend & Forecast", expanded=False):
        ts = df.set_index("Date")[metric].resample("M").sum().rename(metric)
        ts = ts.to_frame().reset_index()
        ts["MA"] = ts[metric].rolling(ma_window).mean()

        fig_ts = px.line(ts, x="Date", y=[metric, "MA"],
                         title=f"{metric} Trend (MA={ma_window}mo)")
        fig_ts.update_traces(selector={"name": "MA"}, line_dash="dash")
        st.plotly_chart(fig_ts, use_container_width=True)

        if metric == "Revenue" and len(ts) >= 2:
            df_prop = ts.rename(columns={"Date": "ds", metric: "y"})[["ds","y"]].dropna()
            fc = fit_prophet(df_prop, periods=horizon, freq="M")
            fig_fc = px.line(fc, x="ds", y="yhat", title="Forecast")
            fig_fc.add_scatter(x=fc.ds, y=fc.yhat_upper, mode="lines",
                               line_dash="dash", name="Upper")
            fig_fc.add_scatter(x=fc.ds, y=fc.yhat_lower, mode="lines",
                               line_dash="dash", name="Lower")
            st.plotly_chart(fig_fc, use_container_width=True)
    st.markdown("---")

    # Seasonality
    with st.expander("📊 Seasonality Heatmap", expanded=False):
        heat = seasonality_heatmap_data(df, "Date", metric)
        display_seasonality_heatmap(heat, f"Seasonality ({metric})", key="prod_seasonality")
    st.markdown("---")

    # ABC & Volume vs Margin
    with st.expander("🅰️🅱️🅾️ ABC & Revenue Vs Margin", expanded=False):
        abc = summary.nlargest(top_n, metric).assign(
            CumPct=lambda d: d[metric].cumsum() / d[metric].sum() * 100
        )
        abc["Class"] = pd.cut(abc.CumPct, [0,80,95,100], labels=["A","B","C"])
        a_col, b_col = st.columns(2)
        a_col.plotly_chart(
            px.bar(abc, x="ProductName", y="CumPct", color="Class",
                   title="ABC Classification"), use_container_width=True
        )
        share = abc.groupby("Class", as_index=False)[metric].sum()
        b_col.plotly_chart(
            px.pie(share, names="Class", values=metric, hole=0.4,
                   title="Share by Class"), use_container_width=True
        )
    st.markdown("---")

    # Clustering
    with st.expander("🔍 Clustering", expanded=False):
        cluster_df = summary.nlargest(top_n, metric).copy()
        cluster_df["Cluster"] = cluster_products(cluster_df, metric)
        st.plotly_chart(
            px.scatter(cluster_df, x=metric, y="Units", size="Profit",
                       color="Cluster", hover_data=["ProductName"],
                       title="Clusters"), use_container_width=True
        )
    st.markdown("---")

    # Download
    st.download_button(
        "📥 Download Product Data",
        data=summary.to_csv(index=False),
        file_name="products.csv",
        mime="text/csv"
    )
