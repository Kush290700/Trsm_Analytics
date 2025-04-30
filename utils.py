import os
import streamlit as st
import pandas as pd
import requests
from prophet import Prophet
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import numpy as np
import calendar

pd.set_option("styler.render.max_elements", 500_000)

# ──────────────────────────────────────────────────────────────────────────────
# DATE FILTER HELPER
# ──────────────────────────────────────────────────────────────────────────────
def filter_by_date(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Pure function: filter df.Date between two Timestamps (inclusive)."""
    return df[(df.Date >= start_date) & (df.Date <= end_date)]

# ──────────────────────────────────────────────────────────────────────────────
# FORECAST HELPER
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def fit_prophet(df: pd.DataFrame, periods: int = 12, freq: str = "ME") -> pd.DataFrame:
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df)
    fut = m.make_future_dataframe(periods=periods, freq=freq)
    return m.predict(fut)

# ──────────────────────────────────────────────────────────────────────────────
# SEASONALITY HEATMAP DATA & DISPLAY
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def seasonality_heatmap_data(
    df: pd.DataFrame, date_col: str, val_col: str
) -> pd.DataFrame:
    # 1) Aggregate by month‐end
    tmp = (
        df
        .groupby(pd.Grouper(key=date_col, freq="ME"))[val_col]
        .sum()
        .reset_index()
    )
    # 2) Extract month name and year
    tmp["MonthNum"] = tmp[date_col].dt.month
    tmp["Month"]    = tmp["MonthNum"].apply(lambda m: calendar.month_abbr[m])
    tmp["Year"]     = tmp[date_col].dt.year.astype(str)

    # 3) Make Month a categorical with the correct order
    month_order = list(calendar.month_abbr)[1:]  # ['Jan','Feb',…,'Dec']
    tmp["Month"] = pd.Categorical(tmp["Month"], categories=month_order, ordered=True)

    # 4) Pivot into a matrix Year × Month
    pivot = (
        tmp
        .pivot(index="Month", columns="Year", values=val_col)
        .fillna(0)
        .reindex(month_order)     # ensures all months appear in order
    )
    return pivot

def display_seasonality_heatmap(
    pivot: pd.DataFrame, title: str, key: str
) -> None:
    fig = px.imshow(
        pivot,
        text_auto=".0f",
        aspect="auto",
        title=title,
        labels={"x": "Year", "y": "Month", "color": title}
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

# ──────────────────────────────────────────────────────────────────────────────
# RFM SCATTER
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def rfm_scatter(df: pd.DataFrame, key: str) -> None:
    now = df.Date.max()
    rfm = (
        df.groupby("CustomerName")
          .agg(
            Recency   = ("Date",    lambda x: (now - x.max()).days),
            Frequency = ("OrderId", "nunique"),
            Monetary  = ("Revenue", "sum")
          )
          .reset_index()
    )
    rfm["R"] = pd.qcut(rfm.Recency, 4, labels=[4,3,2,1]).astype(int)
    rfm["F"] = pd.qcut(rfm.Frequency, 4, labels=[1,2,3,4]).astype(int)
    rfm["M"] = pd.qcut(rfm.Monetary, 4, labels=[1,2,3,4]).astype(int)
    rfm["Segment"] = rfm.R.map(str) + rfm.F.map(str) + rfm.M.map(str)

    fig = px.scatter(
        rfm,
        x="Recency", y="Monetary",
        size="Frequency", color="Segment",
        hover_data=["CustomerName"],
        title="RFM Segmentation"
    )
    fig.update_layout(xaxis_title="Recency (days)", yaxis_title="Monetary ($)")
    st.plotly_chart(fig, use_container_width=True, key=key)

# ──────────────────────────────────────────────────────────────────────────────
# SUPPLIER HELPERS
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def get_supplier_summary(df: pd.DataFrame) -> pd.DataFrame:
    sup = (
        df.groupby("SupplierName")
          .agg(
            TotalRev  = ("Revenue","sum"),
            TotalProf = ("Profit",  "sum"),
            Orders    = ("OrderId","nunique")
          )
          .reset_index()
    )
    sup["MarginPct"] = sup.TotalProf / sup.TotalRev * 100
    return sup

@st.cache_data
def get_monthly_supplier(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby([pd.Grouper(key="Date", freq="ME"), "SupplierName"])["Revenue"]
          .sum()
          .reset_index()
    )

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOMER HELPERS
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_interpurchase(df: pd.DataFrame) -> pd.Series:
    """
    Compute days between successive orders across all customers.
    """
    diffs = (
        df.sort_values(["CustomerName","Date"])
          .groupby("CustomerName")["Date"]
          .diff()
          .dt.days
          .dropna()
    )
    return diffs

# ──────────────────────────────────────────────────────────────────────────────
# PRODUCT VOLATILITY (mean, std, CV) — fills NaNs before plotting
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_volatility(
    df: pd.DataFrame,
    metric: str,
    period: str = "ME"
) -> pd.DataFrame:
    """
    Compute per‐product mean, std and coefficient of variation (CV),
    aggregating `metric` over each freq‐period (default month‐end "ME").
    """
    # 1) build periodized time series
    ts = (
        df
        .groupby([pd.Grouper(key="Date", freq=period), "ProductName"])[metric]
        .sum()
        .reset_index()
    )
    # 2) aggregate
    stats = (
        ts
        .groupby("ProductName")[metric]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    # 3) fill any NaN std (e.g. single‐point series)
    stats["std"] = stats["std"].fillna(0.0)
    # 4) CV = std/mean (guard divide‐by‐zero)
    stats["CV"] = stats.apply(
        lambda row: (row["std"] / row["mean"]) if row["mean"] else 0.0,
        axis=1
    )
    return stats

@st.cache_data(show_spinner=False)
def get_supplier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize supplier‐level Rev/Cost/Profit and compute margin.
    """
    sup = (
        df.groupby("SupplierName")
          .agg(
            TotalRev   = ("Revenue", "sum"),
            TotalCost  = ("Cost",    "sum"),
            TotalProf  = ("Profit",  "sum"),
            Orders     = ("OrderId", "nunique"),
          )
          .reset_index()
    )
    sup["MarginPct"] = np.where(
        sup.TotalRev > 0,
        sup.TotalProf / sup.TotalRev * 100,
        0.0
    )
    return sup

@st.cache_data(show_spinner=False)
def get_monthly_supplier(df: pd.DataFrame, metric: str = "Revenue") -> pd.DataFrame:
    """
    Month‐by‐month totals for any supplier metric (Revenue, Cost, or Profit).
    """
    return (
        df.groupby([pd.Grouper(key="Date", freq="M"), "SupplierName"])[metric]
          .sum()
          .reset_index()
    )
# ──────────────────────────────────────────────────────────────────────────────
# LABOR DATA FETCHER
# ──────────────────────────────────────────────────────────────────────────────

