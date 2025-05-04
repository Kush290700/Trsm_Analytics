# tabs/trend.py
import streamlit as st
import pandas as pd
import plotly.express as px

from utils import fit_prophet, seasonality_heatmap_data, display_seasonality_heatmap

def render(df: pd.DataFrame):
    st.subheader("📊 Sales Trend & Forecast")

    # — Sidebar settings only —
    with st.sidebar.expander("🔧 Trend Settings", expanded=True):
        metric = st.selectbox("Metric", ["Revenue","ShippedWeightLb"], key="t1_metric")
        gran   = st.selectbox("Granularity", ["Daily","Weekly","Monthly","Quarterly"], key="t1_gran")
        freq_map = {"Daily":"D","Weekly":"W","Monthly":"M","Quarterly":"Q"}
        freq = freq_map[gran]
        ma_window = st.slider("Moving-avg window", 1, 30, 7, key="t1_ma")
        horizon   = st.slider("Forecast horizon", 1, 52, 12, key="t1_hor")

    col = metric

    # build time-series & MA
    ts = (
        df.set_index("Date")[col]
          .resample(freq)
          .sum()
          .reset_index()
    )
    ts[col] = ts[col].clip(lower=0)
    ts["MA"] = ts[col].rolling(ma_window, min_periods=1).mean().clip(lower=0)

    fig_tr = px.line(
        ts, x="Date", y=[col,"MA"],
        labels={"value":metric,"variable":""},
        title=f"{metric} Trend ({gran})",
        template="plotly_white"
    )
    fig_tr.update_traces(selector={"name":"MA"}, line=dict(dash="dash"))
    st.plotly_chart(fig_tr, use_container_width=True)

    # MoM & YoY
    if len(ts) >= 2:
        latest,prev = ts.iloc[-1][col],ts.iloc[-2][col]
        mom = (latest-prev)/prev*100 if prev else 0
        c1,c2,c3 = st.columns(3)
        c1.metric(f"{gran} {metric}", f"{latest:,.0f}")
        c2.metric("MoM Δ", f"{mom:+.1f}%")
        if gran in ["Monthly","Quarterly"] and len(ts) > (12 if gran=="Monthly" else 4):
            ref = ts.iloc[-(12 if gran=="Monthly" else 4)][col]
            yoy = (latest-ref)/ref*100 if ref else 0
            c3.metric("YoY Δ", f"{yoy:+.1f}%")
    st.markdown("---")

    # Yearly totals
    yearly = ts.assign(Year=ts.Date.dt.year).groupby("Year")[col].sum().reset_index()
    fig_year = px.bar(yearly, x="Year", y=col, text_auto=",.0f", title=f"Yearly {metric}", template="plotly_white")
    st.plotly_chart(fig_year, use_container_width=True)

    # Cumulative
    ts["Cumulative"] = ts[col].cumsum()
    fig_cum = px.area(ts, x="Date", y="Cumulative", title=f"Cumulative {metric}", template="plotly_white")
    st.plotly_chart(fig_cum, use_container_width=True)
    st.markdown("---")

    # Seasonality heatmap
    heat = seasonality_heatmap_data(df, "Date", col)
    display_seasonality_heatmap(heat, f"Seasonality ({metric})", key="t1_heat")
    st.markdown("---")

    # Forecast
    if len(ts) >= 2:
        dfp = ts.rename(columns={"Date":"ds",col:"y"})[["ds","y"]]
        fc = fit_prophet(dfp, periods=horizon, freq=freq)
        for c in ["yhat","yhat_lower","yhat_upper"]:
            fc[c] = fc[c].clip(lower=0)
        fig_fc = px.line(fc, x="ds", y="yhat", title=f"{metric} Forecast (+{horizon})", template="plotly_white")
        fig_fc.add_scatter(x=fc.ds, y=fc.yhat_upper, mode="lines", line=dict(dash="dash"), name="Upper")
        fig_fc.add_scatter(x=fc.ds, y=fc.yhat_lower, mode="lines", line=dict(dash="dash"), name="Lower")
        st.plotly_chart(fig_fc, use_container_width=True, key="t1_fc")
    else:
        st.info("Need ≥2 points to forecast.")
