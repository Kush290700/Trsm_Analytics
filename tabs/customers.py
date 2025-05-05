# File: tabs/customers.py
import streamlit as st
import pandas as pd
import plotly.express as px

from utils import (
    filter_by_date,
    compute_interpurchase,
    compute_rfm,
    compute_cohort_retention,
    seasonality_heatmap_data,
    display_seasonality_heatmap
)

@st.cache_data
def cluster_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """Add KMeans cluster labels to RFM DataFrame."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    X = rfm[['Recency', 'Frequency', 'Monetary']].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)
    rfm['Cluster'] = KMeans(n_clusters=4, random_state=42).fit_predict(X_scaled).astype(str)
    return rfm


def render(df: pd.DataFrame):
    st.subheader("👥 Customer Intelligence")

    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    # Sidebar filters
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    with st.sidebar.expander("🔧 Customer Filters", expanded=True):
        date_range = st.date_input(
            "Date Range", [min_date, max_date], key="cust_date"
        )
        regions = ["All"] + sorted(df['RegionName'].dropna().unique())
        products = ["All"] + sorted(df['ProductName'].dropna().unique())
        sel_regs = st.multiselect(
            "Regions", regions, default=["All"], key="cust_regs"
        )
        sel_prods = st.multiselect(
            "Products", products, default=["All"], key="cust_prods"
        )

    # Apply filters
    start = pd.to_datetime(date_range[0])
    end = pd.to_datetime(date_range[1])
    dfc = filter_by_date(df, start, end)
    if 'All' not in sel_regs:
        dfc = dfc[dfc['RegionName'].isin(sel_regs)]
    if 'All' not in sel_prods:
        dfc = dfc[dfc['ProductName'].isin(sel_prods)]
    if dfc.empty:
        st.warning("⚠️ No data for those filters.")
        return

    # KPI cards
    total_cust = dfc['CustomerName'].nunique()
    total_rev = dfc['Revenue'].sum()
    total_ord = dfc['OrderId'].nunique()
    avg_order = total_rev / total_ord if total_ord else 0
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers", f"{total_cust:,}")
    k2.metric("Total Revenue", f"${total_rev:,.0f}")
    k3.metric("Total Orders", f"{total_ord:,}")
    k4.metric("Avg Order Value", f"${avg_order:,.2f}")
    st.markdown("---")

    # New / Active / Churn
    with st.expander("📊 New / Active / Churn", expanded=False):
        dfc['Month'] = dfc['Date'].values.astype('datetime64[M]')
        active = (
            dfc.groupby('Month')['CustomerName']
               .nunique()
               .reset_index(name='Active')
        )
        first = (
            dfc.groupby('CustomerName')['Month']
               .min()
               .reset_index(name='FirstMonth')
        )
        new = (
            first.groupby('FirstMonth')['CustomerName']
                 .nunique()
                 .reset_index(name='New')
                 .rename(columns={'FirstMonth': 'Month'})
        )
        summary_mon = (
            active.merge(new, on='Month', how='left')
                  .fillna({'New': 0})
        )
        summary_mon['Cumulative'] = summary_mon['New'].cumsum()

        cohorts = dfc.groupby('Month')['CustomerName'].agg(set)
        months_list = list(cohorts.index)
        churn_list = []
        for prev, curr in zip(months_list, months_list[1:]):
            p = cohorts[prev]
            c = cohorts[curr]
            rate = 100 * (1 - len(p & c) / len(p)) if p else None
            churn_list.append({'Month': curr, 'ChurnRate': rate})
        churn_df = pd.DataFrame(churn_list)

        col1, col2 = st.columns(2)
        col1.plotly_chart(
            px.bar(
                summary_mon, x='Month', y=['New', 'Active'],
                title='New vs Active Customers'
            ), use_container_width=True
        )
        col2.plotly_chart(
            px.line(
                churn_df, x='Month', y='ChurnRate',
                title='Monthly Churn Rate (%)'
            ), use_container_width=True
        )
    st.markdown("---")

    # CLV & Inter-purchase
    with st.expander("💰 CLV & Inter-purchase", expanded=False):
        clv = (
            dfc.groupby('CustomerName')['Revenue']
               .sum()
               .reset_index(name='CLV')
        )
        diffs = compute_interpurchase(dfc)
        c1, c2 = st.columns(2)
        c1.plotly_chart(
            px.histogram(
                clv, x='CLV', nbins=30, marginal='box',
                title='Customer Lifetime Value'
            ), use_container_width=True
        )
        if not diffs.empty:
            c2.plotly_chart(
                px.histogram(
                    diffs, x=diffs.name, nbins=30,
                    marginal='violin',
                    title='Inter-purchase Interval (days)'
                ), use_container_width=True
            )
    st.markdown("---")

    # RFM & Clustering
    with st.expander("👥 RFM & Clustering", expanded=False):
        rfm = compute_rfm(dfc)
        st.plotly_chart(
            px.scatter(
                rfm, x='Recency', y='Monetary', size='Frequency',
                color='RFM', hover_name='CustomerName',
                title='RFM Segmentation'
            ), use_container_width=True
        )
        rfm = cluster_rfm(rfm)
        st.plotly_chart(
            px.scatter(
                rfm, x='Recency', y='Frequency', size='Monetary',
                color='Cluster', hover_name='CustomerName',
                title='RFM Clusters'
            ), use_container_width=True
        )
    st.markdown("---")

    # Cohort Retention
    with st.expander("📈 Cohort Retention", expanded=False):
        retention = compute_cohort_retention(dfc)
        display_seasonality_heatmap(
            retention, title='Customer Cohort Retention', key='cust_cohort'
        )
    st.markdown("---")

    # Download
    st.download_button(
        "📥 Download Filtered Customer Data",
        data=dfc.to_csv(index=False),
        file_name='customers_filtered.csv',
        mime='text/csv'
    )
