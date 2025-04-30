# File: tabs/suppliers.py

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
    fit_prophet
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

@st.cache_data
def compute_volatility(df: pd.DataFrame, metric: str = "Revenue", freq: str = "M") -> pd.DataFrame:
    """
    Compute mean, std, and coefficient of variation of a supplier metric over time.
    """
    ts = (
        df
        .groupby([pd.Grouper(key="Date", freq=freq), "SupplierName"])[metric]
        .sum()
        .reset_index()
    )
    stats = ts.groupby("SupplierName")[metric].agg(mean="mean", std="std").reset_index()
    stats["CV"] = stats["std"] / stats["mean"].replace(0, pd.NA)
    return stats

def render(df: pd.DataFrame):
    st.subheader("🏭 Supplier Analysis")

    # ───────────────────────────────────────────────────────────────
    # Sidebar filters
    # ───────────────────────────────────────────────────────────────
    with st.sidebar.expander("🔧 Suppliers Filters", expanded=True):
        drange = st.date_input(
            "Date range",
            [df.Date.min().date(), df.Date.max().date()],
            key="sup_date"
        )
        sup_list = sorted(df.SupplierName.dropna().unique())
        sel_sup = st.multiselect(
            "Suppliers", ["All"] + sup_list,
            default=["All"], key="sup_sel"
        )
        metric = st.selectbox(
            "Metric", ["Revenue", "Cost", "Profit"],
            index=0, key="sup_metric"
        )
        top_n = st.slider(
            "Top N suppliers", 5, 50, 10, key="sup_topn"
        )
        ma = st.slider(
            "MA window (months)", 1, 12, 3, key="sup_ma"
        )
        hor = st.slider(
            "Forecast horizon (months)", 1, 24, 12, key="sup_hor"
        )

    # unpack dates
    if isinstance(drange, (list, tuple)) and len(drange) == 2:
        start_d, end_d = pd.to_datetime(drange[0]), pd.to_datetime(drange[1])
    else:
        start_d = end_d = pd.to_datetime(drange)

    # filter data
    dfs = filter_by_date(df, start_d, end_d)
    if "All" not in sel_sup:
        dfs = dfs[dfs.SupplierName.isin(sel_sup)]
    if dfs.empty:
        st.warning("⚠️ No data for those filters.")
        return

    # map metric → summary column
    col_map = {"Revenue":"TotalRev", "Cost":"TotalCost", "Profit":"TotalProf"}
    agg_col = col_map[metric]

    # ───────────────────────────────────────────────────────────────
    # KPI cards
    # ───────────────────────────────────────────────────────────────
    summ = get_supplier_summary(dfs)
    total_sup  = summ.SupplierName.nunique()
    total_met  = summ[agg_col].sum()
    total_prof = summ.TotalProf.sum()
    avg_margin = (total_prof / summ.TotalRev.sum() * 100) if summ.TotalRev.sum() else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Suppliers",       f"{total_sup:,}")
    c2.metric(f"Total {metric}", f"${total_met:,.0f}")
    c3.metric("Total Profit",    f"${total_prof:,.0f}")
    c4.metric("Avg Margin %",    f"{avg_margin:.1f}%")
    c5.metric("Data Points",     f"{len(dfs):,}")
    st.markdown("---")

    # ───────────────────────────────────────────────────────────────
    # Distribution histograms
    # ───────────────────────────────────────────────────────────────
    fig_dist1 = px.histogram(
        summ, x=agg_col, nbins=30, marginal="box",
        title=f"{metric} Distribution",
        labels={agg_col: f"{metric} ($)"}
    )
    fig_dist2 = px.histogram(
        summ, x="MarginPct", nbins=30, marginal="violin",
        title="Margin % Distribution",
        labels={"MarginPct":"Margin (%)"}
    )
    st.plotly_chart(fig_dist1, use_container_width=True)
    st.plotly_chart(fig_dist2, use_container_width=True)
    st.markdown("---")

    # ───────────────────────────────────────────────────────────────
    # Top-N bar
    # ───────────────────────────────────────────────────────────────
    topn = summ.nlargest(top_n, agg_col)
    fig_top = px.bar(
        topn,
        x=agg_col, y="SupplierName",
        orientation="h", text_auto=",.0f",
        title=f"Top {top_n} Suppliers by {metric}",
        labels={agg_col: metric, "SupplierName":"Supplier"}
    )
    st.plotly_chart(fig_top, use_container_width=True)
    st.markdown("---")

    # ───────────────────────────────────────────────────────────────
    # Time-series trend + MA + Forecast
    # ───────────────────────────────────────────────────────────────
    ts = (
        dfs.set_index("Date")[metric]
           .resample("M")
           .sum()
           .reset_index()
    )
    ts["MA"] = ts[metric].rolling(ma).mean()

    fig_tr = px.line(
        ts, x="Date", y=[metric, "MA"],
        labels={"value":metric,"variable":""},
        title=f"{metric} Trend (MA={ma}mo)"
    )
    fig_tr.update_traces(selector=dict(name="MA"), line_dash="dash")
    st.plotly_chart(fig_tr, use_container_width=True)

    if len(ts) >= 2:
        dfp = ts.rename(columns={"Date":"ds", metric:"y"}).dropna()
        fore = fit_prophet(dfp, periods=hor, freq="M")
        fig_fc = px.line(fore, x="ds", y="yhat", title=f"{metric} Forecast (+{hor}mo)")
        fig_fc.add_scatter(x=fore.ds, y=fore.yhat_upper, mode="lines",
                           line_dash="dash", name="Upper")
        fig_fc.add_scatter(x=fore.ds, y=fore.yhat_lower, mode="lines",
                           line_dash="dash", name="Lower")
        st.plotly_chart(fig_fc, use_container_width=True)
    st.markdown("---")

    # ───────────────────────────────────────────────────────────────
    # Treemaps
    # ───────────────────────────────────────────────────────────────
    for path, title in [
        (["RegionName","SupplierName","ProductName"],  f"{metric} by Region→Supplier→Product"),
        (["RegionName","SupplierName","CustomerName"], f"{metric} by Region→Supplier→Customer")
    ]:
        treedf = dfs.groupby(path)[metric].sum().reset_index()
        fig_tm = px.treemap(treedf, path=path, values=metric, title=title)
        st.plotly_chart(fig_tm, use_container_width=True)
        st.markdown("---")

    # ───────────────────────────────────────────────────────────────
    # Scatter + drill-down
    # ───────────────────────────────────────────────────────────────
    fig_sc = px.scatter(
        summ, x=agg_col, y="MarginPct", size="Orders", hover_name="SupplierName",
        title=f"{metric} vs Margin %"
    )
    st.plotly_chart(fig_sc, use_container_width=True, key="sup_scatter")
    clicked = plotly_events(fig_sc, click_event=True, key="sup_click")

    if clicked:
        sup = clicked[0].get("hovertext") or topn.iloc[clicked[0]["pointIndex"]]["SupplierName"]
        st.markdown(f"#### Details for **{sup}**")
        dfp = dfs[dfs.SupplierName == sup]
        prod = (
            dfp.groupby("ProductName")
               .agg(
                   Revenue=("Revenue","sum"),
                   Cost   =("Cost",   "sum"),
                   Profit =("Profit","sum"),
                   Orders =("OrderId","nunique")
               )
               .reset_index()
               .sort_values(agg_col, ascending=False)
        )
        st.dataframe(prod, use_container_width=True)
        st.markdown("---")

    # ───────────────────────────────────────────────────────────────
    # Volatility analysis
    # ───────────────────────────────────────────────────────────────
    vol = compute_volatility(dfs, metric=metric, freq="M")
    # drop any suppliers where std/mean/CV is NaN or mean is zero
    vol = vol.dropna(subset=["mean", "std", "CV"])
    vol = vol[vol["mean"] > 0]

    fig_vol = px.scatter(
        vol,
        x="mean",
        y="CV",
        size="std",
        hover_name="SupplierName",
        title=f"{metric} Volatility (mean vs CV)",
        labels={"mean": f"Avg {metric}", "CV": "Coeff of Variation", "std": "Std Dev"},
        template="plotly_white",
    )
    st.plotly_chart(fig_vol, use_container_width=True)
    st.markdown("---")

    # ───────────────────────────────────────────────────────────────
    # K-Means clustering on Top-N
    # ───────────────────────────────────────────────────────────────
    X = StandardScaler().fit_transform(topn[[agg_col,"Orders","MarginPct"]])
    topn["Cluster"] = KMeans(n_clusters=4, random_state=42).fit_predict(X).astype(str)
    fig_cl = px.scatter(
        topn, x=agg_col, y="MarginPct", size="Orders", color="Cluster",
        hover_data=["SupplierName"], title="Clusters on Top Suppliers"
    )
    st.plotly_chart(fig_cl, use_container_width=True)
    st.markdown("---")

    # ───────────────────────────────────────────────────────────────
    # Seasonality heatmap
    # ───────────────────────────────────────────────────────────────
    heat = seasonality_heatmap_data(dfs, "Date", metric)
    display_seasonality_heatmap(heat, f"Seasonality ({metric})", key="sup_season")
    st.markdown("---")

    # ───────────────────────────────────────────────────────────────
    # Drill-down table
    # ───────────────────────────────────────────────────────────────
    st.subheader("🔍 Drill-down Table")
    detail = (
        dfs.groupby(["SupplierName","CustomerName","ProductName"])
           .agg(
             Revenue=("Revenue","sum"),
             Profit=("Profit","sum"),
             Orders=("OrderId","nunique")
           )
           .reset_index()
    )
    st.dataframe(
        detail.style.format({"Revenue":"{:,}","Profit":"{:,}","Orders":"{:,}"}),
        use_container_width=True
    )
