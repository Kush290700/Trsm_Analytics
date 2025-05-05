# File: utils.py
import os
import logging

import streamlit as st
import pandas as pd
import numpy as np
import calendar
import plotly.express as px
from prophet import Prophet

# set up logger for CSV loader & prep
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# PURE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def filter_by_date(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Return rows where df.Date is between start_date and end_date inclusive."""
    return df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

# ──────────────────────────────────────────────────────────────────────────────
# CACHED TRANSFORMS & VISUALS
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def fit_prophet(df: pd.DataFrame, periods: int = 12, freq: str = "M") -> pd.DataFrame:
    """Fit a Prophet model on a DataFrame with columns ['ds','y'], return forecast."""
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df)
    fut = m.make_future_dataframe(periods=periods, freq=freq)
    return m.predict(fut)

@st.cache_data
def seasonality_heatmap_data(df: pd.DataFrame, date_col: str, val_col: str) -> pd.DataFrame:
    """Aggregate val_col by calendar month, pivot into Month×Year for heatmap."""
    tmp = (
        df
        .groupby(pd.Grouper(key=date_col, freq="M"))[val_col]
        .sum()
        .reset_index()
    )
    tmp["MonthNum"] = tmp[date_col].dt.month
    tmp["Month"]    = tmp["MonthNum"].map(lambda m: calendar.month_abbr[m])
    tmp["Year"]     = tmp[date_col].dt.year.astype(str)

    month_order = list(calendar.month_abbr)[1:]
    tmp["Month"] = pd.Categorical(tmp["Month"], categories=month_order, ordered=True)

    pivot = (
        tmp
        .pivot(index="Month", columns="Year", values=val_col)
        .fillna(0)
        .reindex(month_order)
    )
    return pivot


def display_seasonality_heatmap(pivot: pd.DataFrame, title: str, key: str) -> None:
    """Render a Plotly heatmap from a pivoted Month×Year DataFrame."""
    fig = px.imshow(
        pivot,
        text_auto=".0f",
        aspect="auto",
        title=title,
        labels={"x":"Year","y":"Month","color":title}
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

@st.cache_data
def rfm_scatter(df: pd.DataFrame, key: str) -> None:
    """Compute RFM segments and render a scatter plot."""
    now = df["Date"].max()
    rfm = (
        df.groupby("CustomerName")
          .agg(
              Recency   = ("Date", lambda x: (now - x.max()).days),
              Frequency = ("OrderId", "nunique"),
              Monetary  = ("Revenue", "sum")
          )
          .reset_index()
    )
    rfm["R"] = pd.qcut(rfm.Recency,   4, labels=[4,3,2,1]).astype(int)
    rfm["F"] = pd.qcut(rfm.Frequency, 4, labels=[1,2,3,4]).astype(int)
    rfm["M"] = pd.qcut(rfm.Monetary,  4, labels=[1,2,3,4]).astype(int)
    rfm["Segment"] = rfm.R.map(str) + rfm.F.map(str) + rfm.M.map(str)

    fig = px.scatter(
        rfm,
        x="Recency",
        y="Monetary",
        size="Frequency",
        color="Segment",
        hover_data=["CustomerName"],
        title="RFM Segmentation"
    )
    fig.update_layout(xaxis_title="Recency (days)", yaxis_title="Monetary ($)")
    st.plotly_chart(fig, use_container_width=True, key=key)

@st.cache_data
def compute_interpurchase(df: pd.DataFrame) -> pd.Series:
    """Compute days between successive orders for each customer."""
    diffs = (
        df
        .sort_values(["CustomerName","Date"])
        .groupby("CustomerName")["Date"]
        .diff()
        .dt.days
        .dropna()
    )
    return diffs

@st.cache_data
def compute_volatility(
    df: pd.DataFrame,
    metric: str,
    period: str = "M",
    freq: str = None
) -> pd.DataFrame:
    """
    Compute mean, std and CV of `metric` aggregated by
    each calendar period per ProductName.
    """
    use_freq = freq if freq is not None else period
    ts = (
        df
        .groupby([pd.Grouper(key="Date", freq=use_freq), "ProductName"])[metric]
        .sum()
        .reset_index()
    )
    stats = ts.groupby("ProductName")[metric].agg(mean="mean", std="std").reset_index()
    stats["std"].fillna(0.0, inplace=True)
    stats["CV"] = np.where(stats["mean"] > 0, stats["std"] / stats["mean"], 0.0)
    return stats.astype({"mean":"float32","std":"float32","CV":"float32"})

@st.cache_data
def get_supplier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize revenue, cost, profit, orders and margin per supplier."""
    sup = (
        df.groupby("SupplierName")
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
        "TotalRev":"float32",
        "TotalCost":"float32",
        "TotalProf":"float32",
        "Orders":"int32",
        "MarginPct":"float32"
    })

@st.cache_data
def get_monthly_supplier(df: pd.DataFrame, metric: str = "Revenue") -> pd.DataFrame:
    """Month-by-month totals of `metric` per supplier."""
    m = (
        df.groupby([pd.Grouper(key="Date", freq="M"), "SupplierName"])[metric]
          .sum()
          .reset_index()
    )
    m[metric] = pd.to_numeric(m[metric], downcast="float")
    return m

@st.cache_data
def summarize_regions(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Aggregate a primary metric (`col`) plus Orders, Customers, Profit by RegionName.
    """
    agg = df.groupby("RegionName").agg(
        Total     = (col,           "sum"),
        Orders    = ("OrderId",      "nunique"),
        Customers = ("CustomerName","nunique"),
        Profit    = ("Profit",       "sum") if "Profit" in df.columns else (col, "sum")
    ).reset_index()
    agg["AvgOrder"] = agg["Total"] / agg["Orders"].replace(0, np.nan)
    agg["MarginPct"] = np.where(
        "Profit" in agg.columns,
        agg["Profit"] / agg["Total"].replace(0, np.nan) * 100,
        np.nan
    )
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# CSV LOADING & FULL DATA PREPARATION
# ──────────────────────────────────────────────────────────────────────────────

def load_csv_tables(csv_dir: str = "data") -> dict[str, pd.DataFrame]:
    table_names = [
        "orders", "order_lines", "products", "customers",
        "regions", "shippers", "suppliers", "shipping_methods", "packs"
    ]
    raw: dict[str, pd.DataFrame] = {}
    for name in table_names:
        path = os.path.join(csv_dir, f"{name}.csv")
        if os.path.exists(path):
            raw[name] = pd.read_csv(path, low_memory=False)
        else:
            raw[name] = pd.DataFrame()
            logger.warning(f"⚠️ Missing table: {name}.csv")
    return raw


def prepare_full_data(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # 1) Validate core tables
    orders = raw.get("orders", pd.DataFrame()).copy()
    lines  = raw.get("order_lines", pd.DataFrame()).copy()
    if orders.empty:
        raise RuntimeError("Missing or empty 'orders.csv'")
    if lines.empty:
        raise RuntimeError("Missing or empty 'order_lines.csv'")

    # 2) Cast key columns to str
    for df_frame, cols in [
        (orders, ["OrderId","CustomerId","SalesRepId","ShippingMethodRequested"]),
        (lines,  ["OrderLineId","OrderId","ProductId","ShipperId"])
    ]:
        for c in cols:
            if c in df_frame.columns:
                df_frame[c] = df_frame[c].astype(str)
            else:
                raise RuntimeError(f"Expected '{c}' in {df_frame}")

    # 3) Join order_lines ⇄ orders
    df = lines.merge(orders, on="OrderId", how="inner", suffixes=("","_order"))
    logger.info(f"After joining orders+lines: {len(df):,} rows")

    # 4) Merge lookup tables (customers, products, etc.)
    lookups = {
        "customers":       ("CustomerId", ["RegionId","CustomerName","IsRetail"], raw.get("customers")),
        "products":        ("ProductId",  ["SKU","ProductName","UnitOfBillingId","SupplierId"], raw.get("products")),
        "regions":         ("RegionId",   ["RegionName"], raw.get("regions")),
        "shippers":        ("ShipperId",  ["Carrier"], raw.get("shippers")),
        "suppliers":       ("SupplierId", ["SupplierName"], raw.get("suppliers")),
        "shipping_methods":("ShippingMethodRequested", ["ShippingMethodName"], raw.get("shipping_methods")),
    }
    for name, (keycol, wanted, lookup_df) in lookups.items():
        if lookup_df is None or lookup_df.empty:
            logger.warning(f"Skipping merge '{name}'—table missing or empty.")
            continue
        lookup = lookup_df.copy()
        if name=="shipping_methods" and "SMId" in lookup.columns:
            lookup = lookup.rename(columns={"SMId":"ShippingMethodRequested"})
        for c in [keycol] + [c for c in wanted if c in lookup.columns]:
            lookup[c] = lookup[c].astype(str)
        valid = [keycol] + [c for c in wanted if c in lookup.columns]
        sub = lookup[valid].drop_duplicates()
        if keycol not in df.columns:
            logger.warning(f"Key '{keycol}' not in main DF—skipping {name}.")
            continue
        df = df.merge(sub, on=keycol, how="left")
        logger.info(f"After merging '{name}': {len(df):,} rows")

    # 5) Incorporate packs for weight, counts, delivery
    packs = raw.get("packs", pd.DataFrame()).copy()
    if not packs.empty and {"PickedForOrderLine","WeightLb","ItemCount","DeliveryDate"}.issubset(packs.columns):
        packs["OrderLineId"] = packs["PickedForOrderLine"].astype(str)
        psum = (
            packs.groupby("OrderLineId", as_index=False)
                 .agg(WeightLb=("WeightLb","sum"), ItemCount=("ItemCount","sum"), DeliveryDate=("DeliveryDate","max"))
        )
        psum["OrderLineId"] = psum["OrderLineId"].astype(str)
        df = df.merge(psum, on="OrderLineId", how="left")
        df[["WeightLb","ItemCount"]] = df[["WeightLb","ItemCount"]].fillna(0.0)
    else:
        df["WeightLb"]   = 0.0
        df["ItemCount"]  = 0.0
        df["DeliveryDate"] = pd.NaT

    # 6) Numeric conversions
    for col in ["QuantityShipped","SalePrice","UnitCost","WeightLb","ItemCount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 7) Revenue, Cost, Profit logic
    df["Revenue"] = np.where(df.get("UnitOfBillingId","") == "3",
                               df["WeightLb"] * df["SalePrice"],
                               df["ItemCount"] * df["SalePrice"])
    df["Cost"]    = np.where(df.get("UnitOfBillingId","") == "3",
                               df["WeightLb"] * df["UnitCost"],
                               df["ItemCount"] * df["UnitCost"])
    df["Profit"]  = df["Revenue"] - df["Cost"]

    # 8) Exclude production items
    if "IsProduction" in df.columns:
        df["IsProduction"] = pd.to_numeric(df["IsProduction"], errors="coerce").fillna(0).astype(int)
        mask = df["IsProduction"] == 1
        df.loc[mask, ["Revenue","Cost","Profit"]] = 0.0
        logger.info(f"Excluded {mask.sum():,} production rows from margin")

    # 9) Date fields & delivery metrics
    df["Date"]         = pd.to_datetime(df.get("CreatedAt_order"), errors="coerce").dt.normalize()
    df["ShipDate"]     = pd.to_datetime(df.get("ShipDate"), errors="coerce")
    df["DeliveryDate"] = pd.to_datetime(df.get("DeliveryDate"), errors="coerce")
    df["DateExpected"] = pd.to_datetime(df.get("DateExpected"), errors="coerce")

    df["TransitDays"]    = (df["DeliveryDate"] - df["ShipDate"]).dt.days.clip(lower=0)
    df["DeliveryStatus"] = np.where(df["DeliveryDate"] <= df["DateExpected"], "On Time", "Late")

    logger.info(f"✅ Final data prepared: {len(df):,} rows | Total Rev=${df['Revenue'].sum():,.2f}")
    return df
