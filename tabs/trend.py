# File: tabs/trend.py
import streamlit as st
import pandas as pd
import plotly.express as px

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
def get_forecast(dfp: pd.DataFrame, periods: int, freq: str) -> pd.DataFrame:
    """Run Prophet forecast and clip negatives."""
    fc = fit_prophet(dfp, periods=periods, freq=freq)
    for c in ('yhat','yhat_lower','yhat_upper'):
        fc[c] = fc[c].clip(lower=0)
    return fc

@st.cache_data(show_spinner=False)
def get_weekday_pattern(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Compute average value by day of week."""
    df2 = df[['Date', col]].dropna()
    df2['Weekday'] = df2['Date'].dt.day_name()
    avg = df2.groupby('Weekday')[col].mean().reindex([
        'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'
    ]).reset_index()
    return avg

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
        ma_window = st.slider('Moving-avg window', 1, 30, 7)
        horizon = st.slider('Forecast horizon (periods)', 1, 12, 6)

    col = metric

    # Time series + MA
    ts = get_resampled(df, col, freq)
    if ts.empty:
        st.info(f'No data to display for {col}.')
        return
    ts['MA'] = ts[col].rolling(ma_window, min_periods=1).mean().clip(lower=0)
    fig = px.line(ts, x='Date', y=[col,'MA'],
                  labels={'value':col,'variable':''},
                  title=f'{col} Trend ({gran})')
    fig.update_traces(selector={'name':'MA'}, line_dash='dash')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')

    # Seasonality heatmap
    st.markdown('### Seasonality Heatmap')
    heat = seasonality_heatmap_data(df, 'Date', col)
    display_seasonality_heatmap(heat, f'Seasonality ({col})', key='trend_season')

    st.markdown('---')

    # Day-of-week pattern
    st.markdown('### Average by Day of Week')
    wk = get_weekday_pattern(df, col)
    fig_wk = px.bar(wk, x='Weekday', y=col, title=f'Average {col} by Weekday')
    st.plotly_chart(fig_wk, use_container_width=True)

    st.markdown('---')

    # Forecast
    if len(ts) >= 2:
        st.markdown('### Forecast')
        dfp = ts.rename(columns={'Date':'ds', col:'y'})[['ds','y']]
        fc = get_forecast(dfp, periods=horizon, freq=freq)
        fig_fc = px.line(fc, x='ds', y='yhat', title=f'{col} Forecast (+{horizon})')
        fig_fc.add_scatter(x=fc.ds, y=fc.yhat_upper, mode='lines', line_dash='dash', name='Upper')
        fig_fc.add_scatter(x=fc.ds, y=fc.yhat_lower, mode='lines', line_dash='dash', name='Lower')
        st.plotly_chart(fig_fc, use_container_width=True)
    else:
        st.info('Need at least 2 data points to forecast.')
