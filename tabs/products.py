# File: tabs/products.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils import (
    filter_by_date,
    seasonality_heatmap_data,
    display_seasonality_heatmap,
    fit_prophet,
    compute_volatility
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ─── CACHED SUMMARY ────────────────────────────────────────────────────────────

@st.cache_data
def summarize_products(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue, units, and profit by product."""
    return (
        df.groupby("ProductName", as_index=False)
          .agg(
              Revenue=("Revenue", "sum"),
              Units  =("ItemCount", "sum"),
              Profit =("Profit", "sum")
          )
    )

# ─── RENDER FUNCTION ─────────────────────────────────────────────────────────

def render(df: pd.DataFrame):
    st.subheader("📦 Product Analytics")

    # —— Data prep ——
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["Date"], inplace=True)

    # —— Sidebar filters (unique keys) ——
    with st.sidebar.expander("🔧 Products Filters & Settings", expanded=True):
        dr = st.date_input(
            "Date Range",
            [df.Date.min().date(), df.Date.max().date()],
            key="prod_date"
        )
        all_prods = ["All"] + sorted(df.ProductName.dropna().unique())
        sel_prods = st.multiselect(
            "Products", all_prods, default=["All"], key="prod_sel"
        )
        metric = st.selectbox(
            "Primary Metric", ["Revenue", "Units", "Profit"],
            index=0, key="prod_metric"
        )
        top_n = st.slider("Top N Products", 5, 50, 10, key="prod_topn")
        ma_window = st.slider(
            "MA Window (months)", 1, 12, 3, key="prod_ma"
        )
        horizon = st.slider(
            "Forecast Horizon (months)", 1, 24, 12, key="prod_hor"
        )

    # —— Apply filters ——
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
    else:
        start = end = pd.to_datetime(dr)
    dfp = filter_by_date(df, start, end)
    if "All" not in sel_prods:
        dfp = dfp[dfp.ProductName.isin(sel_prods)]
    if dfp.empty:
        st.warning("⚠️ No data for these filters.")
        return

    # —— Top-line KPIs ——
    summary = summarize_products(dfp)
    total_rev  = summary.Revenue.sum()
    total_prof = summary.Profit.sum()
    avg_margin = (total_prof / total_rev * 100) if total_rev else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔢 Products",      f"{summary.shape[0]:,}")
    if "SKU" in dfp:
        c2.metric("📦 SKUs",        f"{dfp.SKU.nunique():,}")
    else:
        c2.write("")
    c3.metric("💰 Revenue",      f"${total_rev:,.0f}")
    c4.metric("🧾 Avg Margin %", f"{avg_margin:.1f}%")
    st.markdown("---")

    # —— Top-N Bar Charts ——
    topn = summary.nlargest(top_n, metric)
    col1, col2 = st.columns(2)
    col1.plotly_chart(
        px.bar(
            topn, x="Revenue", y="ProductName", orientation="h",
            text_auto=",.0f", title=f"Top {top_n} by Revenue"
        ),
        use_container_width=True
    )
    col2.plotly_chart(
        px.bar(
            topn, x="Units", y="ProductName", orientation="h",
            text_auto=",.0f", title=f"Top {top_n} by Units"
        ),
        use_container_width=True
    )
    st.markdown("---")

    # —— Distribution Histogram ——
    fig_dist = px.histogram(
        summary, x=metric, nbins=30, marginal="box",
        title=f"{metric} Distribution"
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    st.markdown("---")

    # —— Time Series + MA + Forecast ——
    ts = (
        dfp.set_index("Date")[metric]
           .resample("M").sum()
           .reset_index(name=metric)
    )
    ts["MA"] = ts[metric].rolling(ma_window).mean()

    fig_ts = px.line(
        ts, x="Date", y=[metric, "MA"],
        labels={"value": metric, "variable": ""},
        title=f"{metric} Trend (MA={ma_window})"
    )
    fig_ts.update_traces(selector=dict(name="MA"), line_dash="dash")
    st.plotly_chart(fig_ts, use_container_width=True)

    if len(ts) >= 2 and metric == "Revenue":
        df_prop = ts.rename(columns={"Date": "ds", metric: "y"}).dropna()
        fc = fit_prophet(df_prop, periods=horizon, freq="M")
        fig_fc = px.line(
            fc, x="ds", y="yhat",
            title=f"{metric} Forecast (+{horizon} mo)"
        )
        fig_fc.add_scatter(
            x=fc.ds, y=fc.yhat_upper, mode="lines", line_dash="dash", name="Upper"
        )
        fig_fc.add_scatter(
            x=fc.ds, y=fc.yhat_lower, mode="lines", line_dash="dash", name="Lower"
        )
        st.plotly_chart(fig_fc, use_container_width=True)
    st.markdown("---")

    # —— Seasonality Heatmap ——
    heat = seasonality_heatmap_data(dfp, "Date", metric)
    display_seasonality_heatmap(
        heat, f"Seasonality ({metric})", key="prod_season"
    )
    st.markdown("---")

    # —— ABC Classification & Share Pie ——
    abc = topn.assign(
        CumPct=topn[metric].cumsum() / topn[metric].sum() * 100
    )
    abc["Class"] = pd.cut(abc.CumPct, bins=[0, 80, 95, 100], labels=["A", "B", "C"])
    a_col, b_col = st.columns(2)
    a_col.plotly_chart(
        px.bar(abc, x="ProductName", y="CumPct", color="Class", title="ABC Classification"),
        use_container_width=True
    )
    share = abc.groupby("Class", as_index=False)[metric].sum()
    b_col.plotly_chart(
        px.pie(share, names="Class", values=metric, hole=0.4, title="Share by Class"),
        use_container_width=True
    )
    st.markdown("---")

    # —— Revenue vs Margin Scatter ——
    vol_df = summary.assign(MarginPct=summary.Profit / summary.Revenue * 100)
    st.plotly_chart(
        px.scatter(
            vol_df, x="Revenue", y="MarginPct", size="Units",
            color="MarginPct", title="Revenue vs Margin %"
        ),
        use_container_width=True
    )
    st.markdown("---")

    # —— Volatility Analysis ——
    vol_stats = compute_volatility(dfp, metric, period="M")
    st.plotly_chart(
        px.scatter(
            vol_stats, x="mean", y="CV", size="std",
            hover_name="ProductName", title="Sales Volatility"
        ),
        use_container_width=True
    )
    st.markdown("---")

    # —— K-Means Clustering ——
    cluster_df = topn.fillna(0)[[metric, "Units", "Profit"]]
    clusters = KMeans(n_clusters=4, random_state=42).fit_predict(
        StandardScaler().fit_transform(cluster_df)
    )
    topn["Cluster"] = clusters.astype(str)
    st.plotly_chart(
        px.scatter(
            topn, x="Revenue", y="Units", size="Profit", color="Cluster",
            hover_data=["ProductName"], title="Product Clusters"
        ),
        use_container_width=True
    )
    st.markdown("---")

    # —— Drill-down Expander ——
    with st.expander("🔍 Drill-down by Product"):
        product_drilldown(dfp)
    st.markdown("---")

    # —— CSV Download ——
    st.download_button(
        "📥 Download Filtered Product Data",
        data=dfp.to_csv(index=False),
        file_name="products_filtered.csv",
        mime="text/csv"
    )


# ─── TOP-LEVEL DRILL-DOWN FUNCTION ─────────────────────────────────────────────

def product_drilldown(dfp: pd.DataFrame):
    """
    Detailed drill-down for a single product.
    """
    st.subheader("🔍 Product Drill-down")

    prods = sorted(dfp.ProductName.dropna().unique())
    choice = st.selectbox("Select Product", ["--"] + prods, key="prod_drill")
    if choice == "--":
        st.info("Please choose a product.")
        return

    pf = dfp[dfp.ProductName == choice].copy()
    pf.dropna(subset=["Date"], inplace=True)

    # Key metrics
    rev   = pf.Revenue.sum()
    units = pf.ItemCount.sum()
    prof  = pf.Profit.sum() if "Profit" in pf else np.nan
    custs = pf.CustomerName.nunique()
    first = pf.Date.min().date()
    last  = pf.Date.max().date()
    span  = max((pf.Date.max() - pf.Date.min()).days, 1)
    freq  = pf.OrderId.nunique() / (span / 30)

    cols = st.columns(5)
    cols[0].metric("Revenue",    f"${rev:,.0f}")
    cols[1].metric("Units Sold", f"{units:,}")
    cols[2].metric("Customers",  f"{custs:,}")
    cols[3].metric("First Sale", f"{first}")
    cols[4].metric("Last Sale",  f"{last}")
    st.markdown(f"**Avg Orders/mo:** {freq:.1f}")
    if not np.isnan(prof):
        st.markdown(f"**Profit:** ${prof:,.0f}")
    st.markdown("---")

    # Monthly trend
    agg = {"Revenue":("Revenue","sum"), "Units":("ItemCount","sum")}
    if "Profit" in pf:
        agg["Profit"] = ("Profit","sum")
    monthly = pf.set_index("Date").resample("M").agg(**agg).reset_index()
    st.plotly_chart(
        px.line(monthly, x="Date", y=list(agg.keys()), title=f"{choice} Monthly Trend"),
        use_container_width=True
    )
    st.markdown("---")

    # Seasonality heatmap
    pf["Mon"] = pf.Date.dt.month_name().str[:3]
    pf["Wd"]  = pf.Date.dt.day_name().str[:3]
    season = (
        pf.groupby(["Wd","Mon"])["Revenue"]
          .sum().reset_index()
          .pivot(index="Wd", columns="Mon", values="Revenue")
          .fillna(0)
    )
    st.plotly_chart(
        px.imshow(season,
                  labels=dict(x="Month", y="Weekday", color="Revenue"),
                  title="Seasonality: Revenue"),
        use_container_width=True
    )
    st.markdown("---")

    # Top customers
    top_c = (
        pf.groupby("CustomerName")
          .agg(Spend=("Revenue","sum"), Orders=("OrderId","nunique"))
          .nlargest(10,"Spend").reset_index()
    )
    st.plotly_chart(
        px.bar(top_c, x="Spend", y="CustomerName", orientation="h",
               title="Top 10 Customers", text_auto=",.0f"),
        use_container_width=True
    )
    st.markdown("---")

    # Co-purchase
    orders = pf.OrderId.unique()
    co = dfp[dfp.OrderId.isin(orders) & (dfp.ProductName != choice)]
    co_top = co.ProductName.value_counts().nlargest(10).reset_index()
    co_top.columns = ["ProductName","Count"]
    st.plotly_chart(
        px.bar(co_top, x="Count", y="ProductName", orientation="h",
               title="Top Co-purchased", text_auto=",.0f"),
        use_container_width=True
    )
    st.markdown("---")

    # Price distribution
    if "UnitPrice" in pf:
        st.plotly_chart(
            px.histogram(pf, x="UnitPrice", nbins=20, title="Unit Price Distribution"),
            use_container_width=True
        )
        st.markdown("---")

    # Correlation matrix
    keys = [k for k in agg if k in monthly]
    if len(keys) > 1:
        corr = monthly[keys].corr()
        st.plotly_chart(
            px.imshow(corr, text_auto=True, title="Metric Correlations"),
            use_container_width=True
        )
    st.markdown("---")

        # —— Optional 6-mo Forecast ——  
    if st.checkbox("Show 6-mo Forecast", key="prod_fc2"):
        from prophet import Prophet
        # Prepare history
        monthly = (
            pf.set_index("Date")
              .resample("M")
              .agg(Revenue=("Revenue","sum"), Units=("ItemCount","sum"))
              .reset_index()
        )
        # Choose the same metric you want to forecast, here "Units"
        hist = monthly.rename(columns={"Date":"ds", "Units":"y"})[["ds","y"]].dropna()
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False)
        m.fit(hist)
        fut = m.make_future_dataframe(periods=6, freq="M")
        pr = m.predict(fut)

        # Merge actuals into forecast frame
        pr_merged = pr[["ds","yhat","yhat_lower","yhat_upper"]].merge(
            hist, on="ds", how="left"
        )

        # Plot both actual (y) and forecast (yhat)
        fig_fc = px.line(
            pr_merged,
            x="ds",
            y=["y", "yhat"],
            labels={"y":"Actual","yhat":"Forecast"},
            title="Forecast vs Actual (Units)"
        )
        fig_fc.add_scatter(
            x=pr_merged.ds,
            y=pr_merged.yhat_upper,
            mode="lines",
            line_dash="dash",
            name="Upper"
        )
        fig_fc.add_scatter(
            x=pr_merged.ds,
            y=pr_merged.yhat_lower,
            mode="lines",
            line_dash="dash",
            name="Lower"
        )
        st.plotly_chart(fig_fc, use_container_width=True)
