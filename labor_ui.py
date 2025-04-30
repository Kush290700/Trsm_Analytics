import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
from utils import fetch_labor_data

def run_labor_dashboard():
    st.header("👷 Labor & Time-Transaction Dashboard")

    # date range picker, default from 2020-01-01 to today
    start, end = st.sidebar.date_input(
        "Labor date range",
        [date(2020,1,1), date.today()],
        key="labor_drange"
    )
    if len((start,end)) != 2:
        st.warning("Please select both start and end dates.")
        return

    st.info("⏳ Fetching labor data…")
    try:
        labor = fetch_labor_data(start, end)
    except Exception as e:
        st.error(f"Failed to fetch labor data: {e}")
        return

    if labor.empty:
        st.info("No labor records in that period.")
        return

    # KPI cards
    total_hours   = labor["PaidHours"].sum()
    total_cost    = labor["DollarAmount"].sum()
    avg_rate      = total_cost / total_hours if total_hours else 0
    emp_count     = labor["EmployeeCode"].nunique()
    dept_count    = labor["DepartmentName"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Hours",      f"{total_hours:,.1f} h")
    c2.metric("Total Labor Cost", f"${total_cost:,.2f}")
    c3.metric("Avg Rate $/h",     f"${avg_rate:,.2f}")
    c4.metric("Employees",        f"{emp_count:,}")

    st.markdown("---")

    # 1) Labor cost by Department
    dept = (labor
            .groupby("DepartmentName")
            .agg(Hours=("PaidHours","sum"), Cost=("DollarAmount","sum"))
            .reset_index()
            .sort_values("Cost", ascending=False))
    fig1 = px.bar(
        dept, x="Cost", y="DepartmentName",
        orientation="h",
        text_auto="$.2f",
        title="Labor Cost by Department",
        labels={"DepartmentName":"Department","Cost":"Cost ($)"},
        template="plotly_white"
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # 2) Monthly Hours & Cost trend
    mon = (labor
           .set_index("ShiftMatchDate")
           .resample("M")[["PaidHours","DollarAmount"]]
           .sum()
           .rename(columns={"PaidHours":"Hours","DollarAmount":"Cost"})
           .reset_index())
    fig2 = px.line(
        mon, x="ShiftMatchDate", y=["Hours","Cost"],
        labels={"value":"Amount","variable":""},
        title="Monthly Labor Hours & Cost",
        template="plotly_white"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # 3) Heatmap: cost by TimeCategory × month
    heat = (labor
            .assign(
                Month=labor["ShiftMatchDate"].dt.strftime("%b"),
                Year =labor["ShiftMatchDate"].dt.year.astype(str)
            )
            .groupby(["Year","Month","TimeCategory"])["DollarAmount"]
            .sum()
            .reset_index())
    pivot = (heat
             .pivot(index="Month", columns="TimeCategory", values="DollarAmount")
             .reindex(["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"])
             .fillna(0))
    fig3 = px.imshow(
        pivot,
        text_auto=".0f",
        aspect="auto",
        title="Labor Cost by Time Category (Heatmap)",
        labels={"x":"TimeCategory","y":"Month","color":"Cost ($)"},
        template="plotly_white"
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # 4) Top employees by cost
    emp = (labor
           .groupby(["EmployeeCode","FirstName","LastName"])
           .agg(Hours=("PaidHours","sum"), Cost=("DollarAmount","sum"))
           .reset_index())
    emp["Employee"] = emp["FirstName"] + " " + emp["LastName"]
    top_emp = emp.nlargest(10, "Cost")
    fig4 = px.bar(
        top_emp, x="Cost", y="Employee",
        orientation="h",
        text_auto="$.2f",
        title="Top 10 Employees by Labor Cost",
        labels={"Employee":"","Cost":"Cost ($)"},
        template="plotly_white"
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # 5) Optional raw data view
    if st.checkbox("Show raw labor transactions"):
        st.dataframe(labor, use_container_width=True)
