# File: tabs/customers.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import filter_by_date, compute_interpurchase, seasonality_heatmap_data, display_seasonality_heatmap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

@st.cache_data
def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    now = df.Date.max()
    agg = (
        df.groupby("CustomerName").agg(
            Recency=("Date", lambda x: (now - x.max()).days),
            Frequency=("OrderId", "nunique"),
            Monetary=("Revenue", "sum"),
        ).reset_index()
    )
    agg["R"] = pd.qcut(agg.Recency, 4, labels=[4,3,2,1]).astype(int)
    agg["F"] = pd.qcut(agg.Frequency, 4, labels=[1,2,3,4]).astype(int)
    agg["M"] = pd.qcut(agg.Monetary, 4, labels=[1,2,3,4]).astype(int)
    agg["RFM"] = agg.R.map(str) + agg.F.map(str) + agg.M.map(str)
    return agg

@st.cache_data
def compute_cohort_retention(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["CohortMonth"] = df2.Date.dt.to_period("M").dt.to_timestamp()
    first = df2.groupby("CustomerName")["CohortMonth"].min().rename("First")
    df2 = df2.join(first, on="CustomerName")
    df2["Period"] = ((df2.CohortMonth.dt.year - df2.First.dt.year) * 12 +
                       (df2.CohortMonth.dt.month - df2.First.dt.month))
    counts = (
        df2.groupby(["First","Period"])["CustomerName"]
           .nunique().reset_index(name="Count")
    )
    sizes = counts[counts.Period == 0].set_index("First")["Count"]
    return (
        counts.pivot(index="First", columns="Period", values="Count")
              .div(sizes, axis=0).fillna(0)
    )

def render(df: pd.DataFrame):
    st.subheader("👥 Customer Intelligence")
    df = df.copy()
    df = filter_by_date(df, df.Date.min(), df.Date.max())

    # Sidebar
    with st.sidebar.expander("🔧 Customer Filters", expanded=True):
        date_range = st.date_input(
            "Date Range", [df.Date.min().date(), df.Date.max().date()], key="cust_date"
        )
        regions = ["All"] + sorted(df.RegionName.dropna().unique())
        products = ["All"] + sorted(df.ProductName.dropna().unique())
        sel_regs = st.multiselect("Regions", regions, default=["All"], key="cust_regs")
        sel_prods = st.multiselect("Products", products, default=["All"], key="cust_prods")

    # Apply filters
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    dfc = filter_by_date(df, start, end)
    if "All" not in sel_regs:
        dfc = dfc[dfc.RegionName.isin(sel_regs)]
    if "All" not in sel_prods:
        dfc = dfc[dfc.ProductName.isin(sel_prods)]
    if dfc.empty:
        st.warning("⚠️ No data for those filters.")
        return

    # KPIs
    total_cust = dfc.CustomerName.nunique()
    total_rev = dfc.Revenue.sum()
    total_ord = dfc.OrderId.nunique()
    avg_order = total_rev / total_ord if total_ord else 0
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers", f"{total_cust:,}")
    k2.metric("Total Revenue", f"${total_rev:,.0f}")
    k3.metric("Total Orders", f"{total_ord:,}")
    k4.metric("Avg Order Value", f"${avg_order:,.2f}")
    st.markdown("---")

    # New / Active / Cumulative
    dfc["Month"] = dfc.Date.dt.to_period("M").dt.to_timestamp()
    active = dfc.groupby("Month")["CustomerName"].nunique().reset_index(name="Active")
    first_order = dfc.groupby("CustomerName")["Month"].min().reset_index(name="FirstMonth")
    new = (first_order.groupby("FirstMonth")["CustomerName"]
               .nunique().reset_index(name="New").rename(columns={"FirstMonth":"Month"}))
    summary_mon = (active.merge(new, on="Month", how="left")
                         .assign(New=lambda d: d.New.fillna(0).astype(int)))
    summary_mon["Cumulative"] = summary_mon.New.cumsum()
    churn_list = []
    sets = dfc.groupby("Month")["CustomerName"].agg(set)
    months = list(sets.index)
    for prev, curr in zip(months, months[1:]):
        p, c = sets[prev], sets[curr]
        rate = 100 * (1 - len(c & p)/len(p)) if p else np.nan
        churn_list.append({"Month": curr, "ChurnRate": rate})
    churn_df = pd.DataFrame(churn_list)

    st.plotly_chart(px.bar(summary_mon, x="Month", y=["New","Active"], title="New vs Active Customers"), use_container_width=True)
    st.plotly_chart(px.line(churn_df, x="Month", y="ChurnRate", title="Monthly Churn Rate (%)"), use_container_width=True)
    st.markdown("---")

    # CLV & Inter-purchase
    clv = dfc.groupby("CustomerName")["Revenue"].sum().reset_index(name="CLV")
    diffs = compute_interpurchase(dfc)
    col1, col2 = st.columns(2)
    col1.plotly_chart(px.histogram(clv, x="CLV", nbins=30, marginal="box", title="Customer Lifetime Value"), use_container_width=True)
    if not diffs.empty:
        col2.plotly_chart(px.histogram(diffs, x=diffs.name, nbins=30, marginal="violin", title="Inter-purchase Interval (days)"), use_container_width=True)
    st.markdown("---")

    # RFM & Clustering
    rfm = compute_rfm(dfc)
    st.plotly_chart(px.scatter(rfm, x="Recency", y="Monetary", size="Frequency", color="RFM", hover_name="CustomerName", title="RFM Segmentation"), use_container_width=True)
    seg_counts = rfm["RFM"].value_counts().rename_axis("RFM").reset_index(name="Count")
    X = StandardScaler().fit_transform(rfm[["Recency","Frequency","Monetary"]])
    rfm["Cluster"] = KMeans(n_clusters=4, random_state=42).fit_predict(X).astype(str)
    st.plotly_chart(px.scatter(rfm, x="Recency", y="Frequency", size="Monetary", color="Cluster", hover_name="CustomerName", title="RFM Clusters"), use_container_width=True)
    st.markdown("---")

    # Cohort retention heatmap
    retention = compute_cohort_retention(dfc)
    display_seasonality_heatmap(retention, title="Customer Cohort Retention", key="cust_cohort")
    st.markdown("---")

    # CSV download
    st.download_button(
        "📥 Download Filtered Customer Data",
        data=dfc.to_csv(index=False),
        file_name="customers_filtered.csv",
        mime="text/csv"
    )
