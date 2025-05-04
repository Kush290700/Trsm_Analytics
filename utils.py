# utils.py
import streamlit as st
import pandas as pd
import numpy as np
import calendar
import plotly.express as px
from prophet import Prophet

def filter_by_date(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    return df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

@st.cache_data
def fit_prophet(df: pd.DataFrame, periods: int = 12, freq: str = "M") -> pd.DataFrame:
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df)
    fut = m.make_future_dataframe(periods=periods, freq=freq)
    return m.predict(fut)

@st.cache_data
def seasonality_heatmap_data(df: pd.DataFrame, date_col: str, val_col: str) -> pd.DataFrame:
    tmp = (
        df
        .groupby(pd.Grouper(key=date_col, freq="M"))[val_col]
        .sum()
        .reset_index()
    )
    tmp["Month"] = tmp[date_col].dt.month.map(lambda m: calendar.month_abbr[m])
    tmp["Year"]  = tmp[date_col].dt.year.astype(str)
    month_order = list(calendar.month_abbr)[1:]
    tmp["Month"] = pd.Categorical(tmp["Month"], categories=month_order, ordered=True)
    return (
        tmp
        .pivot(index="Month", columns="Year", values=val_col)
        .fillna(0)
        .reindex(month_order)
    )

def display_seasonality_heatmap(pivot: pd.DataFrame, title: str, key: str) -> None:
    fig = px.imshow(pivot, text_auto=".0f", aspect="auto", title=title,
                    labels={"x": "Year", "y": "Month", "color": title})
    st.plotly_chart(fig, use_container_width=True, key=key)

@st.cache_data
def compute_interpurchase(df: pd.DataFrame) -> pd.Series:
    return (
        df
        .sort_values(["CustomerName", "Date"])
        .groupby("CustomerName")["Date"]
        .diff()
        .dt.days
        .dropna()
    )

@st.cache_data
def compute_volatility(df: pd.DataFrame, metric: str, *, freq: str = None,
                       period: str = None, group_col: str = "ProductName") -> pd.DataFrame:
    _freq = freq or period or "M"
    ts = (
        df
        .groupby([pd.Grouper(key="Date", freq=_freq), group_col])[metric]
        .sum()
        .reset_index()
    )
    stats = (
        ts
        .groupby(group_col)[metric]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    stats["std"] = stats["std"].fillna(0.0)
    stats["CV"] = np.where(stats["mean"] > 0, stats["std"] / stats["mean"], 0.0)
    return stats.astype({
        group_col: "string",
        "mean": "float32",
        "std": "float32",
        "CV": "float32"
    })

@st.cache_data
def get_supplier_summary(df: pd.DataFrame) -> pd.DataFrame:
    sup = (
        df
        .groupby("SupplierName")
        .agg(
            TotalRev  = ("Revenue", "sum"),
            TotalCost = ("Cost",    "sum"),
            TotalProf = ("Profit",  "sum"),
            Orders    = ("OrderId", "nunique")
        )
        .reset_index()
    )
    sup["MarginPct"] = np.where(
        sup["TotalRev"] > 0,
        sup["TotalProf"] / sup["TotalRev"] * 100,
        0.0
    )
    return sup.astype({
        "SupplierName": "string",
        "TotalRev": "float32",
        "TotalCost": "float32",
        "TotalProf": "float32",
        "Orders": "int32",
        "MarginPct": "float32"
    })

@st.cache_data
def get_monthly_supplier(df: pd.DataFrame, metric: str = "Revenue") -> pd.DataFrame:
    m = (
        df
        .groupby([pd.Grouper(key="Date", freq="M"), "SupplierName"])[metric]
        .sum()
        .reset_index()
    )
    m[metric] = pd.to_numeric(m[metric], downcast="float")
    return m
