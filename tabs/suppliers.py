# tabs/suppliers.py
import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events

from utils import fit_prophet, seasonality_heatmap_data, display_seasonality_heatmap, compute_volatility, get_supplier_summary

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def render(df: pd.DataFrame):
    st.subheader("🏭 Supplier Analysis")

    df = df.copy()

    # — Sidebar settings only —
    with st.sidebar.expander("🔧 Supplier Settings", expanded=True):
        sup_list = sorted(df.SupplierName.dropna().unique())
        sel_sup = st.multiselect("Suppliers", ["All"]+sup_list, default=["All"], key="sup_sel")
        metric = st.selectbox("Metric", ["Revenue","Cost","Profit"], key="sup_metric")
        top_n  = st.slider("Top N suppliers",5,50,10,key="sup_topn")
        ma     = st.slider("MA window (mo)",1,12,3,key="sup_ma")
        hor    = st.slider("Forecast horiz (mo)",1,24,12,key="sup_hor")

    if "All" not in sel_sup:
        df = df[df.SupplierName.isin(sel_sup)]
    if df.empty:
        st.warning("No data after filters."); return

    col_map = {"Revenue":"TotalRev","Cost":"TotalCost","Profit":"TotalProf"}
    agg_col = col_map[metric]

    summ = get_supplier_summary(df)
    total_sup  = summ.SupplierName.nunique()
    total_met  = summ[agg_col].sum()
    total_prof = summ.TotalProf.sum()
    avg_margin = total_prof/summ.TotalRev.sum()*100 if summ.TotalRev.sum() else 0

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Suppliers", f"{total_sup:,}")
    c2.metric(f"Total {metric}", f"${total_met:,.0f}")
    c3.metric("Total Profit", f"${total_prof:,.0f}")
    c4.metric("Avg Margin %", f"{avg_margin:.1f}%")
    c5.metric("Data Points", f"{len(df):,}")
    st.markdown("---")

    # Distribution
    fig1 = px.histogram(summ, x=agg_col, nbins=30, marginal="box", title=f"{metric} Distribution")
    fig2 = px.histogram(summ, x="MarginPct", nbins=30, marginal="violin", title="Margin % Distribution")
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("---")

    # Top-N
    topn = summ.nlargest(top_n, agg_col)
    st.plotly_chart(px.bar(topn, x=agg_col, y="SupplierName", orientation="h", text_auto=",.0f",
                           title=f"Top {top_n} Suppliers by {metric}"), use_container_width=True)
    st.markdown("---")

    # Trend + MA + Forecast
    ts = df.set_index("Date")[metric].resample("M").sum().reset_index()
    ts["MA"] = ts[metric].rolling(ma).mean()
    fig_tr = px.line(ts, x="Date", y=[metric,"MA"], labels={"value":metric}, title=f"{metric} Trend")
    fig_tr.update_traces(selector={"name":"MA"}, line_dash="dash")
    st.plotly_chart(fig_tr, use_container_width=True)
    if len(ts)>=2:
        fc = fit_prophet(ts.rename(columns={"Date":"ds",metric:"y"})[["ds","y"]], periods=hor, freq="M")
        fig_fc = px.line(fc, x="ds", y="yhat", title="Forecast")
        fig_fc.add_scatter(x=fc.ds, y=fc.yhat_upper, mode="lines", line_dash="dash", name="Upper")
        fig_fc.add_scatter(x=fc.ds, y=fc.yhat_lower, mode="lines", line_dash="dash", name="Lower")
        st.plotly_chart(fig_fc, use_container_width=True)
    st.markdown("---")

    # Treemaps
    for path,title in [
        (["RegionName","SupplierName","ProductName"],  f"{metric} by Region→Supplier→Product"),
        (["RegionName","SupplierName","CustomerName"], f"{metric} by Region→Supplier→Customer")
    ]:
        treedf = df.groupby(path)[metric].sum().reset_index()
        st.plotly_chart(px.treemap(treedf, path=path, values=metric, title=title), use_container_width=True)
        st.markdown("---")

    # Scatter + drill
    fig_sc = px.scatter(summ, x=agg_col, y="MarginPct", size="Orders", hover_name="SupplierName",
                        title=f"{metric} vs Margin %")
    st.plotly_chart(fig_sc, use_container_width=True, key="sup_scatter")
    clicked = plotly_events(fig_sc, click_event=True, key="sup_click")
    if clicked:
        sup = clicked[0].get("hovertext") or topn.iloc[clicked[0]["pointIndex"]]["SupplierName"]
        st.markdown(f"#### Details for **{sup}**")
        sub = df[df.SupplierName==sup]
        prod = (sub.groupby("ProductName").agg(
                    Revenue=("Revenue","sum"),
                    Cost   =("Cost","sum"),
                    Profit =("Profit","sum"),
                    Orders =("OrderId","nunique")
                ).reset_index().sort_values(agg_col,ascending=False))
        st.dataframe(prod, use_container_width=True)
        st.markdown("---")

    # Volatility
    vol = compute_volatility(df, metric, freq="M").dropna().query("mean>0")
    st.plotly_chart(px.scatter(vol, x="mean", y="CV", size="std", hover_name="SupplierName",
                               title="Volatility"), use_container_width=True)
    st.markdown("---")

    # Clustering
    X = StandardScaler().fit_transform(topn[[agg_col,"Orders","MarginPct"]])
    topn["Cluster"] = KMeans(n_clusters=4, random_state=42).fit_predict(X).astype(str)
    st.plotly_chart(px.scatter(topn, x=agg_col, y="MarginPct", size="Orders", color="Cluster",
                               hover_data=["SupplierName"], title="Clusters"), use_container_width=True)
    st.markdown("---")

    # Seasonality
    heat = seasonality_heatmap_data(df, "Date", metric)
    display_seasonality_heatmap(heat, f"Seasonality ({metric})", key="sup_season")
    st.markdown("---")

    # Drill-down table
    st.subheader("🔍 Drill-down Table")
    detail = df.groupby(["SupplierName","CustomerName","ProductName"]).agg(
        Revenue=("Revenue","sum"),
        Profit=("Profit","sum"),
        Orders=("OrderId","nunique")
    ).reset_index()
    st.dataframe(detail, use_container_width=True)

    st.download_button("📥 Download Supplier Data", data=detail.to_csv(index=False),
                       file_name="suppliers.csv", mime="text/csv")
