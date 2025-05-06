# File: tabs/trend.py
import streamlit as st
import pandas as pd
import plotly.express as px
import calendar

from utils import fit_prophet, seasonality_heatmap_data, display_seasonality_heatmap

@st.cache_data(show_spinner=False)
def get_resampled(df: pd.DataFrame, col: str, freq: str) -> pd.DataFrame:
    """Resample the metric time series and clip negatives."""
    ts = df.set_index("Date")[col].resample(freq).sum().reset_index()
    ts[col] = ts[col].clip(lower=0)
    return ts

@st.cache_data(show_spinner=False)
def get_yearly_monthly(ts: pd.DataFrame, col: str) -> pd.DataFrame:
    """Aggregate the resampled series into Year–Month buckets for stacking."""
    df2 = ts.copy()
    df2['Year'] = df2['Date'].dt.year
    df2['Month'] = df2['Date'].dt.month.map(lambda m: calendar.month_abbr[m])
    # ensure correct month order
    month_order = list(calendar.month_abbr)[1:]
    df2['Month'] = pd.Categorical(df2['Month'], categories=month_order, ordered=True)
    monthly = (
        df2.groupby(['Year','Month'])[col]
           .sum()
           .reset_index()
    )
    return monthly

@st.cache_data(show_spinner=False)
def get_forecast(dfp: pd.DataFrame, periods: int, freq: str) -> pd.DataFrame:
    """Run Prophet forecast and clip negatives."""
    fc = fit_prophet(dfp, periods=periods, freq=freq)
    for c in ("yhat","yhat_lower","yhat_upper"):
        fc[c] = fc[c].clip(lower=0)
    return fc


def render(df: pd.DataFrame):
    st.subheader("📊 Sales Trend & Forecast")

    # Sidebar settings only
    with st.sidebar.expander("🔧 Trend Settings", expanded=True):
        metric    = st.selectbox("Metric", ["Revenue","ShippedWeightLb"], key="t1_metric")
        gran      = st.selectbox(
            "Granularity",
            ["Daily","Weekly","Monthly","Quarterly"],
            index=2,  # default to Monthly
            key="t1_gran"
        )
        freq_map  = {"Daily":"D","Weekly":"W","Monthly":"M","Quarterly":"Q"}
        freq      = freq_map[gran]
        ma_window = st.slider("Moving-avg window", 1, 30, 7, key="t1_ma")
        horizon   = st.slider("Forecast horizon (periods)", 1, 24, 12, key="t1_hor")

    col = metric

    # Build time series and MA
    ts = get_resampled(df, col, freq)
    ts["MA"] = ts[col].rolling(ma_window, min_periods=1).mean().clip(lower=0)

    fig_tr = px.line(
        ts, x="Date", y=[col, "MA"],
        labels={"value":metric, "variable":""},
        title=f"{metric} Trend ({gran})", template="plotly_white"
    )
    fig_tr.update_traces(selector={"name":"MA"}, line=dict(dash="dash"))
    st.plotly_chart(fig_tr, use_container_width=True)

    # MoM & YoY metrics
    if len(ts) >= 2:
        latest, prev = ts.iloc[-1][col], ts.iloc[-2][col]
        mom = (latest - prev) / prev * 100 if prev else 0
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{gran} {metric}", f"{latest:,.0f}")
        c2.metric("MoM Δ", f"{mom:+.1f}%")
        if gran in ("Monthly","Quarterly"):
            step = 12 if gran == "Monthly" else 4
            if len(ts) > step:
                ref = ts.iloc[-step][col]
                yoy = (latest - ref) / ref * 100 if ref else 0
                c3.metric("YoY Δ", f"{yoy:+.1f}%")
    st.markdown("---")

    # Yearly totals stacked by month
    yearly_monthly = get_yearly_monthly(ts, col)
    fig_year = px.bar(
        yearly_monthly,
        x="Year",
        y=col,
        color="Month",
        text_auto=",.0f",
        title=f"Yearly {metric} (stacked by month)",
        template="plotly_white"
    )
    st.plotly_chart(fig_year, use_container_width=True)

    # Seasonality heatmap
    heat = seasonality_heatmap_data(df, "Date", col)
    display_seasonality_heatmap(heat, f"Seasonality ({metric})", key="t1_heat")
    st.markdown("---")

    # Forecast
    if len(ts) >= 2:
        dfp = ts.rename(columns={"Date":"ds", col:"y"})[["ds","y"]]
        fc = get_forecast(dfp, periods=horizon, freq=freq)
        fig_fc = px.line(
            fc, x="ds", y="yhat", title=f"{metric} Forecast (+{horizon} periods)", template="plotly_white"
        )
        fig_fc.add_scatter(x=fc.ds, y=fc.yhat_upper, mode="lines", line=dict(dash="dash"), name="Upper")
        fig_fc.add_scatter(x=fc.ds, y=fc.yhat_lower, mode="lines", line=dict(dash="dash"), name="Lower")
        st.plotly_chart(fig_fc, use_container_width=True, key="t1_fc")
    else:
        st.info("Need at least 2 data points to forecast.")
