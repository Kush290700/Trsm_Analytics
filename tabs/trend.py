# File: tabs/trend.py

import streamlit as st
import pandas as pd
import plotly.express as px

from utils import (
    filter_by_date,
    fit_prophet,
    seasonality_heatmap_data,
    display_seasonality_heatmap,
)

def render(df: pd.DataFrame):
    st.subheader("📊 Sales Trend & Forecast")

    # ───────────────────────────────────────────────────────────────
    # Sidebar Filters
    # ───────────────────────────────────────────────────────────────
    with st.sidebar.expander("🔧 Trend Filters", expanded=True):
        # Metric
        metric = st.selectbox(
            "Metric",
            ["Revenue", "ShippedWeightLb"],
            key="t1_metric"
        )
        col = metric

        # Granularity
        granularity = st.selectbox(
            "Granularity",
            ["Daily", "Weekly", "Monthly", "Quarterly"],
            key="t1_gran"
        )
        freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q"}
        freq = freq_map[granularity]

        # Date range
        drange = st.date_input(
            "Date Range",
            [df.Date.min().date(), df.Date.max().date()],
            min_value=df.Date.min().date(),
            max_value=df.Date.max().date(),
            key="t1_drange"
        )

        # SKU – Product picker
        sku_prod = (
            df[["SKU", "ProductName"]]
            .dropna(subset=["ProductName"])
            .drop_duplicates()
            .sort_values(["SKU", "ProductName"])
        )
        prod_options = ["All"] + [
            f"{row.SKU} – {row.ProductName}"
            for _, row in sku_prod.iterrows()
        ]
        selected = st.multiselect(
            "Product(s)",
            prod_options,
            default=["All"],
            key="t1_prod"
        )

        # Moving‐avg & Forecast horizon
        ma_window = st.slider("Moving-avg window", 1, 30, 7, key="t1_ma")
        horizon   = st.slider("Forecast horizon", 1, 52, 12, key="t1_hor")

    # ───────────────────────────────────────────────────────────────
    # Apply Filters
    # ───────────────────────────────────────────────────────────────
    # Unpack date range
    if isinstance(drange, (list, tuple)) and len(drange) == 2:
        start_date, end_date = drange
    else:
        start_date = end_date = drange
    start_ts = pd.to_datetime(start_date)
    end_ts   = pd.to_datetime(end_date)

    # Filter by date
    df_t = filter_by_date(df, start_ts, end_ts)

    # Filter by product selection
    if "All" not in selected:
        chosen_names = [s.split("–",1)[1].strip() for s in selected]
        df_t = df_t[df_t.ProductName.isin(chosen_names)]

    # ───────────────────────────────────────────────────────────────
    # Time Series & MA (clipped ≥ 0)
    # ───────────────────────────────────────────────────────────────
    ts = (
        df_t
        .set_index("Date")[col]
        .resample(freq)
        .sum()
        .reset_index()
    )
    # never below zero
    ts[col] = ts[col].clip(lower=0)
    ts["MA"] = ts[col].rolling(ma_window, min_periods=1).mean().clip(lower=0)

    fig_trend = px.line(
        ts,
        x="Date",
        y=[col, "MA"],
        labels={"value": metric, "variable": ""},
        title=f"{metric} Trend ({granularity})",
        template="plotly_white"
    )
    fig_trend.update_traces(selector=dict(name="MA"), line=dict(dash="dash"))
    st.plotly_chart(fig_trend, use_container_width=True)

    # ───────────────────────────────────────────────────────────────
    # MoM & YoY Metrics
    # ───────────────────────────────────────────────────────────────
    if len(ts) >= 2:
        latest = ts.iloc[-1][col]
        prev   = ts.iloc[-2][col]
        mom = (latest - prev) / prev * 100 if prev else 0

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{granularity} {metric}", f"{latest:,.0f}")
        c2.metric("MoM Δ", f"{mom:+.1f}%")

        if granularity in ["Monthly", "Quarterly"]:
            lag = 12 if freq == "ME" else 4
            if len(ts) > lag:
                ref = ts.iloc[-lag][col]
                yoy = (latest - ref) / ref * 100 if ref else 0
                c3.metric("YoY Δ", f"{yoy:+.1f}%")

    st.markdown("---")

    # ───────────────────────────────────────────────────────────────
    # Yearly Totals
    # ───────────────────────────────────────────────────────────────
    yearly = (
        ts
        .assign(Year=ts.Date.dt.year)
        .groupby("Year")[col]
        .sum()
        .reset_index()
    )
    fig_year = px.bar(
        yearly,
        x="Year",
        y=col,
        text_auto=",.0f",
        title=f"Yearly {metric}",
        labels={"Year": "Year", col: metric},
        template="plotly_white"
    )
    st.plotly_chart(fig_year, use_container_width=True)

    # ───────────────────────────────────────────────────────────────
    # Cumulative
    # ───────────────────────────────────────────────────────────────
    ts["Cumulative"] = ts[col].cumsum()
    fig_cum = px.area(
        ts,
        x="Date",
        y="Cumulative",
        title=f"Cumulative {metric}",
        labels={"Cumulative": f"Cumulative {metric}"},
        template="plotly_white"
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    st.markdown("---")

    # ───────────────────────────────────────────────────────────────
    # Seasonality Heatmap
    # ───────────────────────────────────────────────────────────────
    heat = seasonality_heatmap_data(df_t, "Date", col)
    display_seasonality_heatmap(
        heat,
        f"Seasonality ({metric})",
        key="t1_heat"
    )

    st.markdown("---")

    # ───────────────────────────────────────────────────────────────
    # Prophet Forecast (clip outputs ≥ 0)
    # ───────────────────────────────────────────────────────────────
    if len(ts) >= 2:
        dfp = ts.rename(columns={"Date":"ds", col:"y"})[["ds","y"]]
        forecast = fit_prophet(dfp, periods=horizon, freq=freq)
        # clamp negative predictions
        for c in ["yhat","yhat_lower","yhat_upper"]:
            forecast[c] = forecast[c].clip(lower=0)

        fig_fc = px.line(
            forecast,
            x="ds", y="yhat",
            labels={"yhat": metric, "ds": "Date"},
            title=f"{metric} Forecast (+{horizon} periods)",
            template="plotly_white"
        )
        fig_fc.add_scatter(
            x=forecast.ds, y=forecast.yhat_upper,
            mode="lines", line=dict(dash="dash"), name="Upper"
        )
        fig_fc.add_scatter(
            x=forecast.ds, y=forecast.yhat_lower,
            mode="lines", line=dict(dash="dash"), name="Lower"
        )
        st.plotly_chart(fig_fc, use_container_width=True, key="t1_fc")
    else:
        st.info("Need ≥2 data points to forecast.")
