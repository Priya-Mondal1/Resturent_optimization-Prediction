




import streamlit as st
import pandas as pd
import plotly.express as px

from utils.header import show_header


st.set_page_config(
    page_title="Restaurant Profit Optimization",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

show_header()

st.title("What If Analysis")

st.write(
    "Simulate different business scenarios and analyze how they affect revenue, cost and profit."
)

st.divider()

st.subheader("Business Inputs")

orders = st.slider(
    "Monthly Orders",
    500,
    10000,
    4000
)

aov = st.slider(
    "Average Order Value",
    5,
    100,
    30
)

commission = st.slider(
    "Commission Rate (%)",
    5,
    35,
    20
)

delivery_cost = st.slider(
    "Delivery Cost per Order",
    1,
    20,
    5
)

growth = st.slider(
    "Growth Factor",
    0.5,
    2.0,
    1.0
)

st.divider()

estimated_revenue = orders * aov * growth

estimated_cost = (
    estimated_revenue * (commission / 100)
) + (orders * delivery_cost)

estimated_profit = estimated_revenue - estimated_cost

profit_margin = (
    estimated_profit / estimated_revenue
) * 100

st.subheader("Performance Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Estimated Revenue",
        f"${estimated_revenue:,.2f}"
    )

with col2:
    st.metric(
        "Estimated Profit",
        f"${estimated_profit:,.2f}"
    )

with col3:
    st.metric(
        "Profit Margin",
        f"{profit_margin:.2f}%"
    )

st.divider()

comparison = pd.DataFrame({
    "Category": [
        "Revenue",
        "Cost",
        "Profit"
    ],
    "Amount": [
        estimated_revenue,
        estimated_cost,
        estimated_profit
    ]
})

fig = px.bar(
    comparison,
    x="Category",
    y="Amount",
    color="Category",
    text="Amount",
    title="Revenue, Cost and Profit Comparison"
)

fig.update_traces(
    texttemplate="$%{text:,.0f}",
    textposition="outside"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

st.subheader("Business Insights")

if commission > 25:
    st.warning(
        "High commission rates reduce profitability."
    )

if delivery_cost > 8:
    st.warning(
        "Delivery costs are high. Optimizing delivery operations may improve profit."
    )

if profit_margin > 35:
    st.success(
        "Excellent projected profit margin."
    )

elif profit_margin > 20:
    st.info(
        "Business performance is healthy."
    )

else:
    st.error(
        "Projected profit margin is low. Consider reducing costs or increasing revenue."
    )

st.divider()

st.subheader("Scenario Report")

report = pd.DataFrame({
    "Metric": [
        "Monthly Orders",
        "Average Order Value",
        "Commission Rate",
        "Delivery Cost",
        "Growth Factor",
        "Estimated Revenue",
        "Estimated Cost",
        "Estimated Profit",
        "Profit Margin"
    ],
    "Value": [
        orders,
        aov,
        f"{commission}%",
        delivery_cost,
        growth,
        round(estimated_revenue, 2),
        round(estimated_cost, 2),
        round(estimated_profit, 2),
        f"{profit_margin:.2f}%"
    ]
})

st.dataframe(
    report,
    use_container_width=True,
    hide_index=True
)

csv = report.to_csv(index=False)

st.download_button(
    "Download What If Analysis Report",
    data=csv,
    file_name="what_if_analysis_report.csv",
    mime="text/csv",
    use_container_width=True
)



st.divider()
st.caption(
    "Restaurant Profit Optimization System | Final Project | 2026"
)