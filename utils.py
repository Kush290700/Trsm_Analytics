# File: utils.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import calendar
import plotly.express as px
from prophet import Prophet

# ──────────────────────────────────────────────────────────────────────────────
# Filters & Time utilities
# ──────────────────────────────────────────────────────────────────────────────

def filter_by_date(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Return rows with df.Date between start and end inclusive."""
    return df[(df["Date"] >= start) & (df["Date"] <= end)]

# ──────────────────────────────────────────────────────────────────────────────
# Prophet forecasting
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def fit_prophet(df: pd.DataFrame, periods: int = 12, freq: str = "M") -> pd.DataFrame:
    """Fit Prophet on ['ds','y'] and return forecast."""
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    return model.predict(future)

# ──────────────────────────────────────────────────────────────────────────────
# Seasonality heatmap
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def seasonality_heatmap_data(df: pd.DataFrame, date_col: str, val_col: str) -> pd.DataFrame:
    """Pivot monthly sums of val_col into Month×Year DataFrame."""
    tmp = df.groupby(pd.Grouper(key=date_col, freq="M"))[val_col].sum().reset_index()
    tmp["Month"] = tmp[date_col].dt.month.map(lambda m: calendar.month_abbr[m])
    tmp["Year"] = tmp[date_col].dt.year.astype(str)
    months = list(calendar.month_abbr)[1:]
    tmp["Month"] = pd.Categorical(tmp["Month"], categories=months, ordered=True)
    pivot = tmp.pivot(index="Month", columns="Year", values=val_col).fillna(0).reindex(months)
    return pivot

@st.cache_data
def display_seasonality_heatmap(pivot: pd.DataFrame, title: str, key: str) -> None:
    """Render Plotly heatmap for pivoted Month×Year DF."""
    fig = px.imshow(pivot, text_auto=".0f", aspect="auto",
                    title=title, labels={"x":"Year","y":"Month","color":title})
    st.plotly_chart(fig, use_container_width=True, key=key)

# ──────────────────────────────────────────────────────────────────────────────
# Inter-purchase intervals
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def compute_interpurchase(df: pd.DataFrame) -> pd.Series:
    """Return days between successive orders per customer."""
    diffs = df.sort_values(["CustomerName","Date"]) \
             .groupby("CustomerName")["Date"].diff().dt.days
    return diffs.dropna()

# ──────────────────────────────────────────────────────────────────────────────
# RFM & cohort analyses
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary and RFM score."""
    now = df["Date"].max()
    rfm = df.groupby("CustomerName").agg(
        Recency=("Date", lambda x: (now - x.max()).days),
        Frequency=("OrderId", "nunique"),
        Monetary=("Revenue", "sum")
    ).reset_index()
    rfm["R"] = pd.qcut(rfm.Recency, 4, labels=[4,3,2,1]).astype(int)
    rfm["F"] = pd.qcut(rfm.Frequency, 4, labels=[1,2,3,4]).astype(int)
    rfm["M"] = pd.qcut(rfm.Monetary, 4, labels=[1,2,3,4]).astype(int)
    rfm["RFM"] = rfm.R.astype(str) + rfm.F.astype(str) + rfm.M.astype(str)
    return rfm

@st.cache_data
def compute_cohort_retention(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cohort retention rates by month."""
    df2 = df.copy()
    df2["Cohort"] = df2.Date.dt.to_period("M").dt.to_timestamp()
    first = df2.groupby("CustomerName")["Cohort"].min().rename("First")
    df2 = df2.join(first, on="CustomerName")
    df2["Period"] = ((df2.Cohort.dt.year - df2.First.dt.year) * 12 +
                       (df2.Cohort.dt.month - df2.First.dt.month))
    counts = df2.groupby(["First","Period"])["CustomerName"].nunique().reset_index("CustomerName').rename(columns={"CustomerName":"Count"})
    sizes = counts[counts.Period==0].set_index("First")["Count"]
    retention = counts.pivot(index="First", columns="Period", values="Count")
    return retention.div(sizes, axis=0).fillna(0)

# ──────────────────────────────────────────────────────────────────────────────
# Supplier summaries
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def get_supplier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary metrics per supplier."""
    sup = df.groupby("SupplierName").agg(
        TotalRev=("Revenue","sum"),
        TotalCost=("Cost","sum"),
        TotalProf=("Profit","sum"),
        Orders=("OrderId","nunique")
    ).reset_index()
    sup["MarginPct"] = np.where(sup.TotalRev>0, sup.TotalProf/sup.TotalRev*100, 0)
    return sup.astype({"TotalRev":"float32","TotalCost":"float32","TotalProf":"float32","Orders":"int32","MarginPct":"float32"})

@st.cache_data
def get_monthly_supplier(df: pd.DataFrame, metric: str="Revenue") -> pd.DataFrame:
    """Monthly totals of metric per supplier."""
    ts = df.groupby([pd.Grouper(key="Date", freq="M"),"SupplierName"])[metric].sum().reset_index()
    ts[metric] = ts[metric].astype(float)
    return ts

# ──────────────────────────────────────────────────────────────────────────────
# CSV loading & full data prep
# ──────────────────────────────────────────────────────────────────────────────

def load_csv_tables(csv_dir: str="data") -> dict[str, pd.DataFrame]:
    """Load core CSV tables by name."""
    tables = ["orders","order_lines","products","customers","regions","shippers","suppliers","shipping_methods","packs"]
    return {name: pd.read_csv(os.path.join(csv_dir,f"{name}.csv"), low_memory=False) if os.path.exists(os.path.join(csv_dir,f"{name}.csv")) else pd.DataFrame() for name in tables}

@st.cache_data
def prepare_full_data(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join and enrich raw tables into single DataFrame."""
    orders = raw.get("orders",pd.DataFrame())
    lines  = raw.get("order_lines",pd.DataFrame())
    if orders.empty or lines.empty:
        raise RuntimeError("orders or order_lines missing")
    for df, cols in [(orders,["OrderId","CustomerId"]),(lines,["OrderLineId","OrderId","ProductId"])]:
        df[cols] = df[cols].astype(str)
    df = lines.merge(orders, on="OrderId")
    # lookups minimal
    for name, (key, cols) in {
        "customers": ("CustomerId",["CustomerName","RegionId"]),
        "products":  ("ProductId", ["ProductName","UnitOfBillingId","SupplierId"]),
        "regions":   ("RegionId",  ["RegionName"]),
        "suppliers": ("SupplierId",["SupplierName"]),
    }.items():
        lut = raw.get(name)
        if lut is not None and not lut.empty:
            df = df.merge(lut[ [key]+cols ].drop_duplicates(), on=key, how="left")
    # packs simple
    packs = raw.get("packs",pd.DataFrame())
    if not packs.empty and "PickedForOrderLine" in packs:
        packs["OrderLineId"] = packs.PickedForOrderLine.astype(str)
        ps = packs.groupby("OrderLineId").agg(WeightLb=("WeightLb","sum"),ItemCount=("ItemCount","sum")).reset_index()
        df = df.merge(ps, on="OrderLineId", how="left").fillna({"WeightLb":0,"ItemCount":0})
    else:
        df["WeightLb"]=0; df["ItemCount"]=0
    # revenue/cost
    df["SalePrice"] = pd.to_numeric(df.get("SalePrice",0),errors="coerce").fillna(0)
    df["UnitCost"]  = pd.to_numeric(df.get("UnitCost",0),errors="coerce").fillna(0)
    df["Revenue"] = np.where(df.get("UnitOfBillingId")=="3", df.WeightLb*df.SalePrice, df.ItemCount*df.SalePrice)
    df["Cost"]    = np.where(df.get("UnitOfBillingId")=="3", df.WeightLb*df.UnitCost, df.ItemCount*df.UnitCost)
    df["Profit"]  = df.Revenue - df.Cost
    # dates
    df["Date"] = pd.to_datetime(df.get("CreatedAt_order"),errors="coerce").dt.normalize()
    return df
