# File: tabs/trend.py
import streamlit as st
import pandas as pd
import plotly.express as px
import calendar

from utils import fit_prophet, seasonality_heatmap_data, display_seasonality_heatmap

# ─── Cached helpers ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_resampled(df: pd.DataFrame, col: str, freq: str) -> pd.DataFrame:
    """Resample the metric time series and clip negatives."""
    if col not in df.columns:
        return pd.DataFrame({'Date': [], col: []})
    ts = df.set_index('Date')[col].resample(freq).sum().reset_index()
    ts[col] = ts[col].clip(lower=0)
    return ts

@st.cache_data(show_spinner=False)
def get_yearly_monthly(ts: pd.DataFrame, col: str) -> pd.DataFrame:
    """Aggregate the resampled series into Year–Month buckets for stacking."""
    if ts.empty:
        return pd.DataFrame({'Year': [], 'Month': [], col: []})
    df2 = ts.copy()
    df2['Year'] = df2['Date'].dt.year
    df2['Month'] = df2['Date'].dt.month.map(lambda m: calendar.month_abbr[m])
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
    for c in ('yhat','yhat_lower','yhat_upper'):
        fc[c] = fc[c].clip(lower=0)
    return fc

# ─── Render function ─────────────────────────────────────────────────────
def render(df: pd.DataFrame):
    st.subheader('📊 Sales Trend & Forecast')

    # Sidebar settings
    with st.sidebar.expander('🔧 Trend Settings', expanded=True):
        metrics = [m for m in ['Revenue','ShippedWeightLb'] if m in df]
        if not metrics:
            st.error('No valid metrics available.')
            return
        metric = st.selectbox('Metric', metrics)
        gran = st.selectbox('Granularity', ['Daily','Weekly','Monthly'], index=2)
        freq_map = {'Daily':'D','Weekly':'W','Monthly':'M'}
        freq = freq_map[gran]
        ma_window = st.slider('Moving-avg window (periods)', 1, 30, 7)
        horizon = st.slider('Forecast horizon (periods)', 1, 12, 6)

    col = metric

    # Trend + MA
    ts = get_resampled(df, col, freq)
    if ts.empty:
        st.info(f'No data to display for {col}.')
        return
    ts['MA'] = ts[col].rolling(ma_window, min_periods=1).mean().clip(lower=0)
    fig_tr = px.line(
        ts, x='Date', y=[col,'MA'],
        labels={'value':col,'variable':''},
        title=f'{col} Trend ({gran})'
    )
    fig_tr.update_traces(selector={'name':'MA'}, line_dash='dash')
    st.plotly_chart(fig_tr, use_container_width=True)
    st.markdown('---')

    # Seasonality heatmap
    st.markdown('### Seasonality Heatmap')
    heat = seasonality_heatmap_data(df, 'Date', col)
    display_seasonality_heatmap(heat, f'Seasonality ({col})', key='trend_season')
    st.markdown('---')

    # Yearly totals stacked by month
    st.markdown('### Yearly Totals (stacked by month)')
    yearly_monthly = get_yearly_monthly(ts, col)
    if not yearly_monthly.empty:
        fig_year = px.bar(
            yearly_monthly,
            x='Year',
            y=col,
            color='Month',
            text_auto=',.0f',
            title=f'Yearly {col} (stacked by month)'
        )
        st.plotly_chart(fig_year, use_container_width=True)
    else:
        st.info('No annual data to display.')
    st.markdown('---')

    # Forecast
    if len(ts) >= 2:
        st.markdown('### Forecast')
        dfp = ts.rename(columns={'Date':'ds', col:'y'})[['ds','y']]
        fc = get_forecast(dfp, periods=horizon, freq=freq)
        fig_fc = px.line(
            fc, x='ds', y='yhat', title=f'{col} Forecast (+{horizon})'
        )
        fig_fc.add_scatter(x=fc.ds, y=fc.yhat_upper, mode='lines', line_dash='dash', name='Upper')
        fig_fc.add_scatter(x=fc.ds, y=fc.yhat_lower, mode='lines', line_dash='dash', name='Lower')
        st.plotly_chart(fig_fc, use_container_width=True)
    else:
        st.info('Need at least 2 data points to forecast.')
