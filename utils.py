# File: utils.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import calendar
import plotly.express as px
from prophet import Prophet

# ──────────────────────────────────────────────────────────────────────────────
# Date filtering
# ──────────────────────────────────────────────────────────────────────────────

def filter_by_date(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Return rows where df.Date is between start and end inclusive."""
    return df[(df["Date"] >= start) & (df["Date"] <= end)]

# ──────────────────────────────────────────────────────────────────────────────
# Prophet forecasting
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def fit_prophet(df: pd.DataFrame, periods: int = 12, freq: str = "M") -> pd.DataFrame:
    """Fit a Prophet model on a DataFrame with ['ds','y'], return forecast."""
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    return model.predict(future)

# ──────────────────────────────────────────────────────────────────────────────
# Seasonality heatmap
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def seasonality_heatmap_data(df: pd.DataFrame, date_col: str, val_col: str) -> pd.DataFrame:
    """Aggregate val_col by month and pivot into Month×Year matrix."""
    tmp = df.groupby(pd.Grouper(key=date_col, freq="M"))[val_col].sum().reset_index()
    tmp['Month'] = tmp[date_col].dt.month.map(lambda m: calendar.month_abbr[m])
    tmp['Year'] = tmp[date_col].dt.year.astype(str)
    months = list(calendar.month_abbr)[1:]
    tmp['Month'] = pd.Categorical(tmp['Month'], categories=months, ordered=True)
    pivot = tmp.pivot(index='Month', columns='Year', values=val_col).fillna(0).reindex(months)
    return pivot

@st.cache_data
def display_seasonality_heatmap(pivot: pd.DataFrame, title: str, key: str) -> None:
    """Render a Plotly heatmap from a pivoted Month×Year DataFrame."""
    fig = px.imshow(pivot, text_auto='.0f', aspect='auto', title=title,
                    labels={'x':'Year','y':'Month','color':title})
    st.plotly_chart(fig, use_container_width=True, key=key)

# ──────────────────────────────────────────────────────────────────────────────
# Inter-purchase intervals
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def compute_interpurchase(df: pd.DataFrame) -> pd.Series:
    """Compute days between successive orders for each customer."""
    diffs = df.sort_values(['CustomerName','Date']) \
             .groupby('CustomerName')['Date'].diff().dt.days
    return diffs.dropna()

# ──────────────────────────────────────────────────────────────────────────────
# RFM segmentation
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary, and RFM score for each customer."""
    now = df['Date'].max()
    rfm = df.groupby('CustomerName').agg(
        Recency   = ('Date',    lambda x: (now - x.max()).days),
        Frequency = ('OrderId', 'nunique'),
        Monetary  = ('Revenue', 'sum')
    ).reset_index()
    rfm['R'] = pd.qcut(rfm.Recency,   4, labels=[4,3,2,1]).astype(int)
    rfm['F'] = pd.qcut(rfm.Frequency, 4, labels=[1,2,3,4]).astype(int)
    rfm['M'] = pd.qcut(rfm.Monetary,  4, labels=[1,2,3,4]).astype(int)
    rfm['RFM'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)
    return rfm

# ──────────────────────────────────────────────────────────────────────────────
# Cohort retention
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def compute_cohort_retention(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cohort retention rates by month."""
    df2 = df.copy()
    df2['CohortMonth'] = df2['Date'].dt.to_period('M').dt.to_timestamp()
    first = df2.groupby('CustomerName')['CohortMonth'].min().rename('First')
    df2 = df2.join(first, on='CustomerName')
    df2['Period'] = ((df2['CohortMonth'].dt.year - df2['First'].dt.year) * 12 +
                     (df2['CohortMonth'].dt.month - df2['First'].dt.month))
    counts = df2.groupby(['First','Period'])['CustomerName'] \
                  .nunique().reset_index(name='Count')
    sizes = counts[counts['Period']==0].set_index('First')['Count']
    retention = counts.pivot(index='First', columns='Period', values='Count')
    return retention.div(sizes, axis=0).fillna(0)

# ──────────────────────────────────────────────────────────────────────────────
# Region summaries
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def summarize_regions(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Aggregate Total, Orders, Customers, Profit by RegionName."""
    agg = df.groupby('RegionName').agg(
        Total     = (col,            'sum'),
        Orders    = ('OrderId',       'nunique'),
        Customers = ('CustomerName',  'nunique'),
        Profit    = ('Profit',        'sum') if 'Profit' in df.columns else (col, 'sum')
    ).reset_index()
    agg['AvgOrder']  = agg['Total'] / agg['Orders'].replace(0, np.nan)
    agg['MarginPct'] = np.where(agg['Total']>0, agg['Profit']/agg['Total']*100, 0)
    return agg.astype({
        'Total':'float32','Orders':'int32','Customers':'int32',
        'Profit':'float32','AvgOrder':'float32','MarginPct':'float32'
    })

# ──────────────────────────────────────────────────────────────────────────────
# Supplier summaries
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def get_supplier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize revenue, cost, profit, and orders per supplier."""
    sup = df.groupby('SupplierName').agg(
        TotalRev  = ('Revenue','sum'),
        TotalCost = ('Cost',   'sum'),
        TotalProf = ('Profit', 'sum'),
        Orders    = ('OrderId','nunique')
    ).reset_index()
    sup['MarginPct'] = np.where(sup.TotalRev>0, sup.TotalProf/sup.TotalRev*100, 0)
    return sup.astype({
        'TotalRev':'float32','TotalCost':'float32',
        'TotalProf':'float32','Orders':'int32','MarginPct':'float32'
    })

@st.cache_data
def get_monthly_supplier(df: pd.DataFrame, metric: str='Revenue') -> pd.DataFrame:
    """Get month-by-month totals of a metric per supplier."""
    ts = df.groupby([pd.Grouper(key='Date', freq='M'),'SupplierName'])[metric].sum().reset_index()
    ts[metric] = ts[metric].astype(float)
    return ts

# ──────────────────────────────────────────────────────────────────────────────
# CSV loading & data preparation
# ──────────────────────────────────────────────────────────────────────────────

def load_csv_tables(csv_dir: str='data') -> dict[str, pd.DataFrame]:
    """Load CSVs for core tables into a dict."""
    names = ['orders','order_lines','products','customers','regions',
             'shippers','suppliers','shipping_methods','packs']
    raw = {}
    for name in names:
        path = os.path.join(csv_dir, f'{name}.csv')
        raw[name] = pd.read_csv(path, low_memory=False) if os.path.exists(path) else pd.DataFrame()
    return raw

@st.cache_data
def prepare_full_data(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join raw CSV tables into a single enriched DataFrame, using correct revenue logic."""
    orders = raw.get('orders',      pd.DataFrame())
    lines  = raw.get('order_lines', pd.DataFrame())
    if orders.empty or lines.empty:
        raise RuntimeError('orders or order_lines missing')

    # Cast keys to str
    orders['OrderId']    = orders['OrderId'].astype(str)
    orders['CustomerId'] = orders['CustomerId'].astype(str)
    lines[['OrderLineId','OrderId','ProductId']] = lines[['OrderLineId','OrderId','ProductId']].astype(str)

    # Merge orders + lines
    df = lines.merge(orders, on='OrderId', how='inner')

    # Lookup merges
    lookups = {
        'customers':         ('CustomerId',               ['CustomerName','RegionId']),
        'products':          ('ProductId',                ['SKU','ProductName','UnitOfBillingId','SupplierId']),
        'regions':           ('RegionId',                 ['RegionName']),
        'suppliers':         ('SupplierId',               ['SupplierName']),
        'shipping_methods':  ('ShippingMethodRequested', ['ShippingMethodName'])
    }
    for name,(key,cols) in lookups.items():
        lut = raw.get(name)
        if lut is None or lut.empty:
            continue
        lut = lut.copy()
        if name == 'shipping_methods':
            if 'SMId' in lut.columns:
                lut.rename(columns={'SMId': key}, inplace=True)
            elif 'ShippingMethodId' in lut.columns:
                lut.rename(columns={'ShippingMethodId': key}, inplace=True)
        if key not in lut.columns or key not in df.columns:
            continue
        lut[key] = lut[key].astype(str)
        df[key] = df[key].astype(str)
        df = df.merge(lut[[key] + cols].drop_duplicates(), on=key, how='left')

    # Packs aggregation
    packs = raw.get('packs', pd.DataFrame())
    if not packs.empty and 'PickedForOrderLine' in packs.columns:
        packs['OrderLineId'] = packs['PickedForOrderLine'].astype(str)
        agg = packs.groupby('OrderLineId').agg(WeightLb=('WeightLb','sum'), ItemCount=('ItemCount','sum')).reset_index()
        agg['OrderLineId'] = agg['OrderLineId'].astype(str)
        df = df.merge(agg, on='OrderLineId', how='left').fillna({'WeightLb':0,'ItemCount':0})
    else:
        df['WeightLb'], df['ItemCount'] = 0.0, 0.0

    # Filter to packed orders only
    if 'OrderStatus' in df.columns:
        df = df[df['OrderStatus']=='packed']

    # Cast UnitOfBillingId to int
    if 'UnitOfBillingId' in df.columns:
        df['UnitOfBillingId'] = pd.to_numeric(df['UnitOfBillingId'], errors='coerce').fillna(0).astype(int)

    # Determine price column for revenue
    price_col = 'Price' if 'Price' in df.columns else ('SalePrice' if 'SalePrice' in df.columns else None)
    if price_col is None:
        raise RuntimeError("Missing Price or SalePrice column for revenue calculation")

    # Compute Revenue using pack weights or item count
    df['Revenue'] = np.where(
        df['UnitOfBillingId'] == 3,
        df['WeightLb'] * df[price_col],
        df['ItemCount'] * df[price_col]
    )

    # Compute Cost & Profit
    if 'UnitCost' in df.columns:
        df['Cost'] = np.where(
            df['UnitOfBillingId'] == 3,
            df['WeightLb'] * df['UnitCost'],
            df['ItemCount'] * df['UnitCost']
        )
    else:
        df['Cost'] = 0.0
    df['Profit'] = df['Revenue'] - df['Cost']

    # Normalize Date
    df['Date'] = pd.to_datetime(df.get('CreatedAt_order'), errors='coerce').dt.normalize()

    # Ensure ShippingMethodName exists
    if 'ShippingMethodName' not in df.columns:
        df['ShippingMethodName'] = np.nan

    return df
    
@st.cache_data(show_spinner=False)
def get_unique(df: pd.DataFrame, col: str) -> list:
    """Return sorted unique non-null values for a given column."""
    if col in df.columns:
        return sorted(df[col].dropna().unique().tolist())
    return []

