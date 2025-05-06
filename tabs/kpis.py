# tabs/kpis.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from utils import seasonality_heatmap_data, display_seasonality_heatmap

@st.cache_data
def compute_monthly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index('Date').resample('ME')['Revenue'].sum().reset_index()

def render(df_all: pd.DataFrame, df: pd.DataFrame):
    st.subheader("🚀 Executive Summary")

    start, end = df.Date.min(), df.Date.max()
    days = (end - start).days + 1

    totals = df.agg({
        'Revenue':'sum','Cost':'sum','Profit':'sum',
        'OrderId':'nunique','CustomerName':'nunique',
        'ItemCount':'sum','WeightLb':'sum'
    }).rename({
        'OrderId':'Orders','CustomerName':'Customers',
        'ItemCount':'Units','WeightLb':'Weight'
    })

    prev = df_all.loc[
        (df_all.Date >= start - pd.Timedelta(days=days)) &
        (df_all.Date <  start)
    ]
    deltas = {k: totals[k] - prev[k].sum() for k in ['Revenue','Cost','Profit']}

    years = end.year - start.year - ((end.month,end.day) < (start.month,start.day))
    if years >= 1:
        yoy_prev = df_all.loc[
            (df_all.Date >= start.replace(year=start.year-1)) &
            (df_all.Date <= end.replace(  year=end.year-1))
        ]
        rev_label = f"{(totals.Revenue/yoy_prev.Revenue.sum()*100 - 100):+.1f}%"
    else:
        rev_label = f"{deltas['Revenue']:+,.0f}"

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Revenue",  f"${totals.Revenue:,.0f}", rev_label)
    c2.metric("Cost",     f"${totals.Cost:,.0f}",    f"{deltas['Cost']:+,.0f}")
    c3.metric("Profit",   f"${totals.Profit:,.0f}",  f"{deltas['Profit']:+,.0f}")
    c4.metric("Orders",   f"{totals.Orders:,}")
    c5.metric("Customers",f"{totals.Customers:,}")
    c6,c7 = st.columns([1,1])
    c6.metric("Units",  f"{int(totals.Units):,}")
    c7.metric("Weight", f"{int(totals.Weight):,} lb")

    st.markdown("---")

    monthly = compute_monthly_revenue(df)
    monthly['MonthLabel'] = monthly['Date'].dt.strftime('%b %Y')
    fig = px.area(
        monthly, x='Date', y='Revenue', line_shape='spline', markers=True,
        hover_data={'MonthLabel':True}, title='Monthly Revenue Trend'
    )
    fig.update_traces(
        fillcolor='rgba(220,38,38,0.3)', line_color='crimson',
        hovertemplate="<b>%{customdata[0]}</b><br>Revenue: $%{y:,.0f}<extra></extra>"
    )
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(buttons=[
                dict(count=1,label='1 mo',step='month',stepmode='backward'),
                dict(count=6,label='6 mo',step='month',stepmode='backward'),
                dict(count=1,label='YTD',step='year',stepmode='todate'),
                dict(count=1,label='1 yr',step='year',stepmode='backward'),
                dict(step='all')
            ]),
            rangeslider=dict(visible=True), type='date', tickformat='%b %Y'
        ),
        yaxis=dict(tickprefix='$', separatethousands=True),
        template='plotly_white', height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    for col,title in [
        ('RegionName','Top 10 Regions by Revenue'),
        ('CustomerName','Top 10 Customers by Revenue'),
        ('ProductName','Top 10 Products by Revenue')
    ]:
        with st.expander(title, expanded=False):
            top = df.groupby(col)['Revenue'].sum().nlargest(10).reset_index()
            fig_t = px.bar(top, x='Revenue', y=col, orientation='h', text_auto=',.0f')
            st.plotly_chart(fig_t, use_container_width=True)
