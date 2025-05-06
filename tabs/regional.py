# File: tabs/regional.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import (
    filter_by_date,
    seasonality_heatmap_data,
    display_seasonality_heatmap,
    fit_prophet,
    summarize_regions,
    get_unique
)
from datetime import datetime

# Cache heavy operations
@st.cache_data(show_spinner=False)
def cached_region_summary(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return summarize_regions(df, col)

@st.cache_data(show_spinner=False)
def cached_resample(df: pd.DataFrame, col: str, freq: str) -> pd.DataFrame:
    return df.groupby(pd.Grouper(key='Date', freq=freq))[col].sum().reset_index()

@st.cache_data(show_spinner=False)
def cached_yoy(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df2 = df.assign(Year=df.Date.dt.year)
    yoy = df2.groupby(['Year','RegionName'])[col].sum().reset_index()
    yoy['YoY%'] = yoy.groupby('RegionName')[col].pct_change() * 100
    return yoy

@st.cache_data(show_spinner=False)
def cached_correlation(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    return df[cols].corr()


def render(df: pd.DataFrame):
    st.subheader("🌎 Regional Analysis")

    # Ensure Date is datetime & drop missing
    df = df.loc[pd.notnull(df['Date'])]

    # Sidebar filters & settings
    with st.sidebar.expander("🔧 Regional Filters & Settings", expanded=True):
        # Date range
        start_date, end_date = st.date_input(
            "Date Range",
            [df.Date.min().date(), df.Date.max().date()],
            min_value=df.Date.min().date(),
            max_value=df.Date.max().date(),
            key='reg_date'
        )
        start = pd.to_datetime(start_date)
        end   = pd.to_datetime(end_date)

        # Metric & granularity
        metric = st.selectbox("Primary Metric", ["Revenue", "ShippedWeightLb", "Orders", "Profit"])
        gran   = st.selectbox("Trend Granularity", ["Monthly","Quarterly","Yearly"])
        freq_map = {"Monthly":"M","Quarterly":"Q","Yearly":"Y"}
        freq  = freq_map[gran]

        # Forecast toggle
        do_fc = metric == "Revenue" and st.checkbox("Enable Forecast", value=False)
        fc_periods = st.slider("Forecast periods (months)", 1, 24, 12) if do_fc else None

        # Dynamic filters
        prods   = ["All"] + get_unique(df, "ProductName")
        methods = ["All"] + get_unique(df, "ShippingMethodName")
        regs    = ["All"] + get_unique(df, "RegionName")
        sel_prods   = st.multiselect("Products", prods, default=["All"])
        sel_methods = st.multiselect("Ship Methods", methods, default=["All"])
        sel_regs    = st.multiselect("Regions", regs, default=["All"])

        show_dist    = st.checkbox("Show Distribution", value=False)
        show_season  = st.checkbox("Show Seasonality", value=True)
        show_corr    = st.checkbox("Show Correlation", value=False)
        show_cluster = st.checkbox("Show Clustering", value=False)

    # Apply filters (cheap in-memory)
    mask = (df.Date >= start) & (df.Date <= end)
    if "All" not in sel_prods:
        mask &= df.ProductName.isin(sel_prods)
    if "All" not in sel_methods:
        mask &= df.ShippingMethodName.isin(sel_methods)
    if "All" not in sel_regs:
        mask &= df.RegionName.isin(sel_regs)
    df_f = df.loc[mask]

    if df_f.empty:
        st.warning("⚠️ No data after filters.")
        return

    # Determine column for aggregation
    col = 'OrderId' if metric == 'Orders' else metric

    # KPI cards
    total_val    = df_f[col].nunique() if metric=='Orders' else df_f[col].sum()
    total_orders = df_f.OrderId.nunique()
    total_cust   = df_f.CustomerName.nunique()
    avg_order    = df_f.Revenue.sum() / total_orders if total_orders else 0
    profit_margin= df_f.Profit.sum()/df_f.Revenue.sum()*100 if 'Profit' in df_f and df_f.Revenue.sum()>0 else np.nan

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric(f"Total {metric}", f"{total_val:,.0f}")
    c2.metric("Orders", f"{total_orders:,}")
    c3.metric("Customers", f"{total_cust:,}")
    c4.metric("Avg Order $", f"{avg_order:,.0f}")
    c5.metric("Profit Margin", f"{profit_margin:.1f}%")
    st.markdown("---")

    ov, det = st.tabs(["📈 Overview","🔍 Details"])

    with ov:
        # Trend
        st.markdown("### Trend")
        ts = cached_resample(df_f, col, freq)
        fig_tr = px.line(ts, x='Date', y=col, title=f"{metric} Trend ({gran})")
        st.plotly_chart(fig_tr, use_container_width=True)

        # Forecast
        if do_fc:
            st.markdown("### Forecast vs Actual")
            dfp = ts.rename(columns={'Date':'ds', col:'y'})
            fc = fit_prophet(dfp, periods=fc_periods, freq=freq)
            merged = fc[['ds','yhat']].merge(dfp, on='ds', how='left')
            fig_fc = px.line(merged, x='ds', y=['y','yhat'],
                             labels={'y':'Actual','yhat':'Forecast'},
                             title="Prophet Forecast")
            st.plotly_chart(fig_fc, use_container_width=True)

        # YoY
        st.markdown("### Year-over-Year % Δ")
        yoy = cached_yoy(df_f, col)
        latest = yoy.Year.max()
        top_yoy = yoy[yoy.Year==latest].nlargest(10,'YoY%')
        fig_yoy = px.bar(top_yoy, x='RegionName', y='YoY%',
                         text=top_yoy['YoY%'].map('{:+.1f}%'.format),
                         title=f"YoY % Δ by Region ({latest})")
        st.plotly_chart(fig_yoy, use_container_width=True)

        # Distribution
        if show_dist:
            st.markdown("### Distribution by Region")
            fig_box = px.box(df_f, x='RegionName', y=col, title=f"{metric} Distribution")
            st.plotly_chart(fig_box, use_container_width=True)

        # Seasonality
        if show_season:
            st.markdown("### Seasonality Heatmap")
            heat = seasonality_heatmap_data(df_f, 'Date', col)
            display_seasonality_heatmap(heat, f"Seasonality — {metric}", key='heat_reg')

        # Correlation
        if show_corr:
            st.markdown("### Correlation Matrix")
            num_cols = [c for c in ['Revenue','ShippedWeightLb','Profit'] if c in df_f]
            if len(num_cols)>1:
                corr = cached_correlation(df_f, num_cols)
                fig_corr = px.imshow(corr, text_auto=True, title="Metric Correlations")
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Not enough numeric metrics for correlation.")

        # Clustering
        if show_cluster:
            st.markdown("### Regional Clustering")
            reg_sum = cached_region_summary(df_f, col)
            X = reg_sum[['Total','Profit']].fillna(0)
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            labels = KMeans(n_clusters=4, random_state=42).fit_predict(StandardScaler().fit_transform(X))
            reg_sum['Cluster'] = labels.astype(str)
            fig_cl = px.scatter(reg_sum, x='Total', y='Profit', size='Total',
                                color='Cluster', hover_name='RegionName',
                                title="Clusters by Total vs Profit")
            st.plotly_chart(fig_cl, use_container_width=True)

    with det:
        st.subheader("Region Drill-down")
        sel_region = st.selectbox("Select Region", regs, index=0)
        df_sub = df_f if sel_region=='All' else df_f[df_f.RegionName==sel_region]
        if df_sub.empty:
            st.info("No data for this region.")
        else:
            reg_sum = cached_region_summary(df_sub, col)
            st.dataframe(reg_sum, use_container_width=True)
            p1, p2 = st.columns(2)
            with p1:
                st.markdown("**Top Products**")
                top_p = df_sub.groupby('ProductName')[col].sum().nlargest(10).reset_index()
                st.plotly_chart(px.bar(top_p, x=col, y='ProductName'), use_container_width=True)
            with p2:
                st.markdown("**Top Customers**")
                top_c = df_sub.groupby('CustomerName')[col].sum().nlargest(10).reset_index()
                st.plotly_chart(px.bar(top_c, x=col, y='CustomerName'), use_container_width=True)

        st.download_button(
            "📥 Download Region Summary",
            data=df_sub.groupby('RegionName')[col].sum().reset_index().to_csv(index=False),
            file_name=f"{sel_region or 'all'}_region_summary.csv",
            mime="text/csv"
        )
